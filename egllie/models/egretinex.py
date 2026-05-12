import torch
from torch import nn
from models.Cross_Restore_RWKV import Cross_Restore_RWKV
import torch.nn.functional as F
from torch.fft import fft2, ifft2
from mmcv.ops import DeformConv2dPack



class AdaptiveGaussianFilter(nn.Module):
    def __init__(self, channels, kernel_size=13, sigma_range=(0.5, 2.0)):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
        self.sigmas = nn.Parameter(torch.rand(channels) * (sigma_range[1] - sigma_range[0]) + sigma_range[0])

    def forward(self, x):
        b, c, h, w = x.shape
        filtered_x = []
        for i in range(c):
            sigma = self.sigmas[i].item()
            kernel = self.create_gaussian_kernel(self.kernel_size, sigma).to(x.device)
            pad_h = max(0, h - self.kernel_size)
            pad_w = max(0, w - self.kernel_size)
            padded_kernel = F.pad(kernel, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
            fft_x = torch.fft.fft2(x[:, i:i+1, :, :])
            fft_kernel = torch.fft.fft2(padded_kernel.unsqueeze(0).unsqueeze(0))
            filtered_fft_x = fft_x * fft_kernel
            filtered_channel = torch.fft.ifft2(filtered_fft_x).real 

            filtered_x.append(filtered_channel)

        return torch.cat(filtered_x, dim=1)

    def create_gaussian_kernel(self, size, sigma):
        ax = torch.arange(-size // 2 + 1., size // 2 + 1.)
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        return kernel

class EISFE(nn.Module):
    def __init__(self, in_channels, mid_channels, kernel_size=5, sigma=1.5):
        super(EISFE, self).__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.conv1x1_a = nn.Conv2d(144, mid_channels, kernel_size=1, bias=False)
        self.conv1x1_b = nn.Conv2d(144, mid_channels, kernel_size=1, bias=False)

        self.ffc = AdaptiveGaussianFilter(mid_channels)

        self.conv3x3 = DeformConv2dPack(mid_channels, mid_channels, kernel_size=3, padding=1)

        self.spatial_attention1 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.spatial_attention2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 16, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(mid_channels // 16, mid_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.output_conv = nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False)

    def forward(self, restored_img_feature, restored_event_feature, img_inp):
        fused_feature = torch.cat([restored_img_feature, restored_event_feature, img_inp], dim=1)

        low_freq = self.conv1x1_a(fused_feature)
        high_freq = self.conv1x1_b(fused_feature)

        low_freq = self.ffc(low_freq)
        high_freq = self.conv3x3(high_freq)

        fused_feature = self.spatial_attention1(low_freq) * low_freq + self.spatial_attention2(high_freq) * high_freq

        fused_feature = fused_feature * self.channel_attention(fused_feature)

        return self.output_conv(fused_feature)

class IllumiinationNet(nn.Module):  # check
    def __init__(self, cfg):
        super().__init__()

        ### illumiantion maps fusion
        self.ill_extractor = nn.Sequential(
            nn.Conv2d(
                cfg.illumiantion_level + 3,
                cfg.illumiantion_level * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                cfg.illumiantion_level * 2,
                cfg.base_chs,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        self.ill_level = cfg.illumiantion_level
        self.illumiantion_set = cfg.illumiantion_set

        self.reduce = nn.Sequential(
            nn.Conv2d(cfg.base_chs, 1, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, batch):
        ### selcet illumiantion map
        ill_list = [int(num) for num in self.illumiantion_set]
        inital_ill = torch.cat(
            [batch["ill_list"][i] for i in ill_list], dim=1
        )
        ### predict inital enhacned illumiantion map
        pred_illu_feature = self.ill_extractor(torch.concat((inital_ill, batch['lowligt_image']), dim=1))  # [B C H W]
        pred_illumaintion = self.reduce(pred_illu_feature)  # [B 1 H W]

        return pred_illumaintion, pred_illu_feature


class ImageEnhanceNet(nn.Module):  # check

    def __init__(self, cfg):
        super().__init__()
        self.base_chs = cfg.base_chs


        self.ev_extractor = nn.Sequential(
            nn.Conv2d(
                cfg.voxel_grid_channel,
                cfg.base_chs * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                cfg.base_chs * 2,
                cfg.base_chs,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        )

        self.img_extractor = nn.Sequential(
            nn.Conv2d(
                3,
                cfg.base_chs * 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(
                cfg.base_chs * 2,
                cfg.base_chs,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )

        self.cross_restore_rwkv = Cross_Restore_RWKV(dim=48)
        self.CrossModalSpectralEnhancer = EISFE(in_channels=48, mid_channels=48)

        self.convtranspose = nn.ConvTranspose2d(
            in_channels=48,
            out_channels=48,
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv = nn.Conv2d(in_channels=48, out_channels=3, kernel_size=1, stride=1, padding=0)
    def forward(self, batch):
        # result of inital enhacnment
        low_light_img = batch["lowligt_image"]  # [B 3 H W]
        pred_illumaintion = batch["illumaintion"]  # [B 1 H W]
        enhance_low_img_mid = low_light_img * pred_illumaintion + low_light_img  # [B 3 H W]

        event_free = batch["event_free"]  # [B 32 H W]

        img_inp = self.img_extractor(enhance_low_img_mid)  # 1 48 128 128
        event_inp = self.ev_extractor(event_free)  # 1 48 128 128

        restored_img_feature, restored_event_feature = self.cross_restore_rwkv(img_inp, event_inp)  # [B C H/2 W/2]

        restored_img_event = self.CrossModalSpectralEnhancer(restored_img_feature,restored_event_feature, img_inp)+restored_img_feature  # [B, 48, 128, 128]
        restored_img_event = self.conv(self.convtranspose(restored_img_event)) 

        enhanced_image = restored_img_event+enhance_low_img_mid
        return enhanced_image


class EgLlie(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.IllumiinationNet = IllumiinationNet(cfg.IlluNet)
        self.ImageEnhanceNet = ImageEnhanceNet(cfg.ImageNet)

    def forward(self, batch):
        batch["illumaintion"], batch['illu_feature'] = self.IllumiinationNet(batch)
        output = self.ImageEnhanceNet(batch)

        outputs = {
            'pred': output,
            'gt': batch["normalligt_image"],
        }

        return outputs

