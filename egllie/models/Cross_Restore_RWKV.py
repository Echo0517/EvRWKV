import math, os
from cycler import K, V
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.fft import fft2, ifft2

T_MAX = 512 * 512

from torch.utils.cpp_extension import load

# Use relative paths for the CUDA extension to ensure cross-platform compatibility
# Adjust the relative path based on the final structure of your GitHub repository
cuda_dir = os.path.join(os.path.dirname(__file__), 'cuda')
wkv_cuda = load(
    name="wkv", 
    sources=[
        os.path.join(cuda_dir, "wkv_op.cpp"),
        os.path.join(cuda_dir, "wkv_cuda.cu")
    ],
    verbose=True,
    extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', f'-DTmax={T_MAX}']
)


class WKV(torch.autograd.Function):
    """
    Custom autograd function for the WKV computation using the compiled CUDA kernel.
    Ensures memory-efficient and fast execution of the time-decaying attention mechanism.
    """
    @staticmethod
    def forward(ctx, B, T, C, w, u, k, v):
        ctx.B = B
        ctx.T = T
        ctx.C = C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0

        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        
        y = torch.empty((B, T, C), device='cuda', memory_format=torch.contiguous_format)
        wkv_cuda.forward(B, T, C, w, u, k, v, y)
        
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        B = ctx.B
        T = ctx.T
        C = ctx.C
        assert T <= T_MAX
        assert B * C % min(C, 1024) == 0
        
        w, u, k, v = ctx.saved_tensors
        gw = torch.zeros((B, C), device='cuda').contiguous()
        gu = torch.zeros((B, C), device='cuda').contiguous()
        gk = torch.zeros((B, T, C), device='cuda').contiguous()
        gv = torch.zeros((B, T, C), device='cuda').contiguous()
        
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        
        wkv_cuda.backward(B, T, C,
                          w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous(),
                          gw, gu, gk, gv)
                          
        if half_mode:
            gw = torch.sum(gw.half(), dim=0)
            gu = torch.sum(gu.half(), dim=0)
            return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            gw = torch.sum(gw.bfloat16(), dim=0)
            gu = torch.sum(gu.bfloat16(), dim=0)
            return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)


def RUN_CUDA(B, T, C, w, u, k, v):
    """Wrapper function to execute the custom WKV CUDA kernel."""
    return WKV.apply(B, T, C, w.cuda(), u.cuda(), k.cuda(), v.cuda())


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        
        # Training phase components
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Inference phase components
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False)
        self.repram_flag = True

    def forward_train(self, x):
        """Multi-branch forward pass for training."""
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)

        out = self.alpha[0] * x + self.alpha[1] * out1x1 + self.alpha[2] * out3x3 + self.alpha[3] * out5x5
        return out

    def reparam_5x5(self):
        """
        Combines the parameters of the identity, 1x1, 3x3, and 5x5 depth-wise 
        convolutions to form a single equivalent 5x5 convolution weight matrix.
        """
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (self.alpha[0] * identity_weight + 
                           self.alpha[1] * padded_weight_1x1 + 
                           self.alpha[2] * padded_weight_3x3 + 
                           self.alpha[3] * self.conv5x5.weight)

        device = self.conv5x5_reparam.weight.device
        combined_weight = combined_weight.to(device)
        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):
        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training == False and self.repram_flag == True:
            # Trigger reparameterization on the first inference step
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            # Use the reparameterized convolution for subsequent inference steps
            out = self.conv5x5_reparam(x)
        return out


class VRWKV_SpatialMix(nn.Module):
    """
    Vision RWKV Spatial Mix Module tailored for dual-stream (Image & Event) fusion.
    Captures global spatial dependencies by formulating the sequence mixing as a 
    linear continuous-time state space model.
    """
    def __init__(self, n_embd, n_layer, layer_id, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        self.recurrence = 2

        # OmniShift for dynamic spatial feature extraction
        self.omni_shift_image = OmniShift(dim=n_embd)
        self.omni_shift_event = OmniShift(dim=n_embd)

        # Image stream parameters
        self.key_image_dw = nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1, groups=n_embd, bias=False)
        self.key_image_pw = nn.Conv2d(in_channels=n_embd, out_channels=attn_sz, kernel_size=1, bias=False)
        self.value_image = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_image = nn.Linear(n_embd, attn_sz, bias=False)

        # Event stream parameters
        self.key_event_dw = nn.Conv2d(in_channels=n_embd, out_channels=n_embd, kernel_size=3, padding=1, groups=n_embd, bias=False)
        self.key_event_pw = nn.Conv2d(in_channels=n_embd, out_channels=attn_sz, kernel_size=1, bias=False)
        self.value_event = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance_event = nn.Linear(n_embd, attn_sz, bias=False)

        self.key_norm = nn.LayerNorm(n_embd) if key_norm else None
        self.output1 = nn.Linear(attn_sz, n_embd, bias=False)
        self.output2 = nn.Linear(attn_sz, n_embd, bias=False)

        # Learnable temporal/spatial decay mechanisms
        with torch.no_grad():
            self.spatial_decay_image = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first_image = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

            self.spatial_decay_event = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))
            self.spatial_first_event = nn.Parameter(torch.randn((self.recurrence, self.n_embd)))

        # Learnable fusion weights
        self.img_w = nn.Parameter(torch.zeros(n_embd))
        self.eve_w = nn.Parameter(torch.zeros(n_embd))

    def jit_func_image(self, x, resolution):
        """Pre-processes the image stream to generate spatial receptance, key, and value."""
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift_image(x)
        
        k_conv = self.key_image_dw(x)  # Depthwise
        k_conv = self.key_image_pw(k_conv)  # Pointwise
        k = rearrange(k_conv, 'b c h w -> b (h w) c')

        x = rearrange(x, 'b c h w -> b (h w) c')
        v = self.value_image(x)
        r = self.receptance_image(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def jit_func_event(self, x, resolution):
        """Pre-processes the event stream to generate spatial receptance, key, and value."""
        h, w = resolution
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift_event(x)
        
        k_conv = self.key_event_dw(x)  
        k_conv = self.key_event_pw(k_conv)  
        k = rearrange(k_conv, 'b c h w -> b (h w) c')
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        v = self.value_event(x)
        r = self.receptance_event(x)
        sr = torch.sigmoid(r)
        return sr, k, v

    def forward(self, x_image, x_event, resolution):
        B, T, C = x_image.size()
        self.device = x_image.device

        sr_image, k_image, v_image = self.jit_func_image(x_image, resolution)
        sr_event, k_event, v_event = self.jit_func_event(x_event, resolution)

        # Cross-modal alternating spatial mixing
        for j in range(self.recurrence):
            if j % 2 == 0:
                k_cross = k_event
                v_image = RUN_CUDA(B, T, C, self.spatial_decay_image[j] / T, self.spatial_first_image[j] / T, k_cross, v_image)

                k_cross_event = k_image
                v_event = RUN_CUDA(B, T, C, self.spatial_decay_event[j] / T, self.spatial_first_event[j] / T, k_cross_event, v_event)
            else:
                h, w = resolution
                
                # Perform orthogonal spatial scanning
                k_cross = rearrange(k_cross, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_image = rearrange(v_image, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_image = RUN_CUDA(B, T, C, self.spatial_decay_image[j] / T, self.spatial_first_image[j] / T, k_cross, v_image)
                k_cross = rearrange(k_cross, 'b (w h) c -> b (h w) c', h=h, w=w)
                v_image = rearrange(v_image, 'b (w h) c -> b (h w) c', h=h, w=w)

                k_cross_event = rearrange(k_cross_event, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_event = rearrange(v_event, 'b (h w) c -> b (w h) c', h=h, w=w)
                v_event = RUN_CUDA(B, T, C, self.spatial_decay_event[j] / T, self.spatial_first_event[j] / T, k_cross_event, v_event)
                k_cross_event = rearrange(k_cross_event, 'b (w h) c -> b (h w) c', h=h, w=w)
                v_event = rearrange(v_event, 'b (w h) c -> b (h w) c', h=h, w=w)

        # Apply learnable adaptive fusion weights
        img_alpha = torch.sigmoid(self.img_w).unsqueeze(0).unsqueeze(1)
        eve_alpha = torch.sigmoid(self.eve_w).unsqueeze(0).unsqueeze(1)

        x1 = x_image * img_alpha + x_event * (1 - img_alpha)
        x2 = x_event * eve_alpha + x_image * (1 - eve_alpha)
  
        x_image = sr_image * x1
        x_event = sr_event * x2

        if self.key_norm is not None:
            x_image = self.key_norm(x_image)
            x_event = self.key_norm(x_event)

        x_image = self.output1(x_image)
        x_event = self.output2(x_event)

        return x_image, x_event


class VRWKV_ChannelMix(nn.Module):
    """
    Channel Mixing module to exchange information across channel dimensions.
    Operates independently on the Image and Event streams utilizing 
    receptance gating mechanisms to control information flow.
    """
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id
        self.n_layer = n_layer
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)

        # Image Stream Components
        self.key_image = nn.Linear(n_embd, hidden_sz, bias=False)
        self.value_image = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance_image = nn.Linear(n_embd, n_embd, bias=False)
        self.omni_shift_image = OmniShift(dim=n_embd)
        self.key_norm_image = nn.LayerNorm(hidden_sz) if key_norm else None

        # Event Stream Components
        self.key_event = nn.Linear(n_embd, hidden_sz, bias=False)
        self.value_event = nn.Linear(hidden_sz, n_embd, bias=False)
        self.receptance_event = nn.Linear(n_embd, n_embd, bias=False)
        self.omni_shift_event = OmniShift(dim=n_embd)
        self.key_norm_event = nn.LayerNorm(hidden_sz) if key_norm else None

    def forward(self, x_image, x_event, resolution):
        h, w = resolution

        # Process Image Stream
        x_image_2d = rearrange(x_image, 'b (h w) c -> b c h w', h=h, w=w)
        x_image_2d = self.omni_shift_image(x_image_2d)
        x_image_shifted = rearrange(x_image_2d, 'b c h w -> b (h w) c')

        k_image = self.key_image(x_image_shifted)
        k_image = torch.square(torch.relu(k_image))
        if self.key_norm_image is not None:
            k_image = self.key_norm_image(k_image)
        kv_image = self.value_image(k_image)

        gate_image = torch.sigmoid(self.receptance_image(x_image_shifted))
        x_image_out = gate_image * kv_image

        # Process Event Stream
        x_event_2d = rearrange(x_event, 'b (h w) c -> b c h w', h=h, w=w)
        x_event_2d = self.omni_shift_event(x_event_2d)
        x_event_shifted = rearrange(x_event_2d, 'b c h w -> b (h w) c')

        k_event = self.key_event(x_event_shifted)
        k_event = torch.square(torch.relu(k_event))
        if self.key_norm_event is not None:
            k_event = self.key_norm_event(k_event)
        kv_event = self.value_event(k_event)

        gate_event = torch.sigmoid(self.receptance_event(x_event_shifted))
        x_event_out = gate_event * kv_event

        return x_image_out, x_event_out


class Block(nn.Module):
    """
    Standard Transformer-like architectural block combining the local feature extraction (DW Conv), 
    global spatial attention mechanism (VRWKV_SpatialMix), and channel integration (VRWKV_ChannelMix).
    """
    def __init__(self, n_embd, n_layer, layer_id, hidden_rate=4, init_mode='fancy', key_norm=False):
        super().__init__()
        self.layer_id = layer_id

        self.dwconv1_image = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.dwconv1_event = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)

        self.dwconv2_image = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)
        self.dwconv2_event = nn.Conv2d(n_embd, n_embd, kernel_size=3, stride=1, padding=1, groups=n_embd, bias=False)

        self.ln1_image = nn.LayerNorm(n_embd)
        self.ln1_event = nn.LayerNorm(n_embd)
        
        self.ln2_image = nn.LayerNorm(n_embd)
        self.ln2_event = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd, n_layer, layer_id, init_mode, key_norm=key_norm)
        self.ffn = VRWKV_ChannelMix(n_embd, n_layer, layer_id, hidden_rate, init_mode, key_norm=key_norm)

        self.gamma1_image = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma1_event = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        
        self.gamma2_image = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma2_event = nn.Parameter(torch.ones(n_embd), requires_grad=True)

    def forward(self, x_image, x_event):
        b, c, h, w = x_image.shape
        resolution = (h, w)

        # 1. Local Spatial Context & Global Spatial Mix
        residual_image = x_image
        x_image = self.dwconv1_image(x_image) + residual_image
        x_image = rearrange(x_image, 'b c h w -> b (h w) c')
        
        residual_event = x_event
        x_event = self.dwconv1_event(x_event) + residual_event
        x_event = rearrange(x_event, 'b c h w -> b (h w) c')

        x_image_att, x_event_att = self.att(self.ln1_image(x_image), self.ln1_event(x_event), resolution)

        x_image = x_image + self.gamma1_image * x_image_att
        x_event = x_event + self.gamma1_event * x_event_att
        
        x_image = rearrange(x_image, 'b (h w) c -> b c h w', h=h, w=w)
        x_event = rearrange(x_event, 'b (h w) c -> b c h w', h=h, w=w)

        # 2. Feed-Forward Network (Channel Mix)
        residual_image = x_image
        x_image = self.dwconv2_image(x_image) + residual_image
        x_image = rearrange(x_image, 'b c h w -> b (h w) c')
        
        residual_event = x_event
        x_event = self.dwconv2_event(x_event) + residual_event
        x_event = rearrange(x_event, 'b c h w -> b (h w) c')

        x_image_ffn, x_event_ffn = self.ffn(self.ln2_image(x_image), self.ln2_event(x_event), resolution)

        x_image = x_image + self.gamma2_image * x_image_ffn
        x_event = x_event + self.gamma2_event * x_event_ffn
        
        x_image = rearrange(x_image, 'b (h w) c -> b c h w', h=h, w=w)
        x_event = rearrange(x_event, 'b (h w) c -> b c h w', h=h, w=w)

        return x_image, x_event


class Downsample(nn.Module):
    """Spatial downsampling module utilizing PixelUnshuffle."""
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    """Spatial upsampling module utilizing PixelShuffle."""
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class TupleBlock(nn.Module):
    """Wrapper class to pass tuple arguments (Image, Event) symmetrically through network blocks."""
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, args):
        x_image, x_event = args
        x_image, x_event = self.block(x_image, x_event)
        return (x_image, x_event)


class Cross_Restore_RWKV(nn.Module):
    """
    Main Hierarchical U-Net Framework orchestrating the entire restoration pipeline.
    Utilizes skip connections and multi-level feature alignment to perform 
    continuous interactive event-guided low-light enhancement.
    """
    def __init__(self, inp_channels=3, out_channels=48, dim=48, num_blocks=[1, 2, 2, 6], num_refinement_blocks=4):
        super(Cross_Restore_RWKV, self).__init__()

        # Encoder Levels
        self.encoder_level1 = nn.Sequential(*[TupleBlock(Block(n_embd=dim, n_layer=num_blocks[0], layer_id=i)) for i in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 2), n_layer=num_blocks[1], layer_id=i)) for i in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim * 2))

        self.encoder_level3 = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 4), n_layer=num_blocks[2], layer_id=i)) for i in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim * 4))

        # Latent Representation Level
        self.latent = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 8), n_layer=num_blocks[3], layer_id=i)) for i in range(num_blocks[3])])

        # Decoder Levels with Channel Reduction for Skip Connections
        self.up4_3 = Upsample(int(dim * 8))
        self.reduce_chan_level3_image = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=True)
        self.reduce_chan_level3_event = nn.Conv2d(int(dim * 8), int(dim * 4), kernel_size=1, bias=True)
        self.decoder_level3 = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 4), n_layer=num_blocks[2], layer_id=i)) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 4))
        self.reduce_chan_level2_image = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=True)
        self.reduce_chan_level2_event = nn.Conv2d(int(dim * 4), int(dim * 2), kernel_size=1, bias=True)
        self.decoder_level2 = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 2), n_layer=num_blocks[1], layer_id=i)) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 2), n_layer=num_blocks[0], layer_id=i)) for i in range(num_blocks[0])])

        # Final Refinement & Output Generation
        self.refinement = nn.Sequential(*[TupleBlock(Block(n_embd=int(dim * 2), n_layer=num_refinement_blocks, layer_id=i)) for i in range(num_refinement_blocks)])

        self.output_img = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.output_eve = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, img_inp, event_inp):
        # Encoder Forward Pass
        out_enc_level1_image, out_enc_level1_event = self.encoder_level1((img_inp, event_inp))

        inp_enc_level2_image = self.down1_2(out_enc_level1_image)
        inp_enc_level2_event = self.down1_2(out_enc_level1_event)
        out_enc_level2_image, out_enc_level2_event = self.encoder_level2((inp_enc_level2_image, inp_enc_level2_event))

        inp_enc_level3_image = self.down2_3(out_enc_level2_image)
        inp_enc_level3_event = self.down2_3(out_enc_level2_event)
        out_enc_level3_image, out_enc_level3_event = self.encoder_level3((inp_enc_level3_image, inp_enc_level3_event))

        inp_enc_level4_image = self.down3_4(out_enc_level3_image)
        inp_enc_level4_event = self.down3_4(out_enc_level3_event)
        latent_image, latent_event = self.latent((inp_enc_level4_image, inp_enc_level4_event))

        # Decoder Forward Pass with Skip Connections
        inp_dec_level3_image = self.up4_3(latent_image)
        inp_dec_level3_event = self.up4_3(latent_event)
        
        inp_dec_level3_image = torch.cat([inp_dec_level3_image, out_enc_level3_image], 1)
        inp_dec_level3_event = torch.cat([inp_dec_level3_event, out_enc_level3_event], 1)
        
        inp_dec_level3_image = self.reduce_chan_level3_image(inp_dec_level3_image)
        inp_dec_level3_event = self.reduce_chan_level3_event(inp_dec_level3_event)
        
        out_dec_level3_image, out_dec_level3_event = self.decoder_level3((inp_dec_level3_image, inp_dec_level3_event))

        inp_dec_level2_image = self.up3_2(out_dec_level3_image)
        inp_dec_level2_event = self.up3_2(out_dec_level3_event)
        
        inp_dec_level2_image = torch.cat([inp_dec_level2_image, out_enc_level2_image], 1)
        inp_dec_level2_event = torch.cat([inp_dec_level2_event, out_enc_level2_event], 1)
        
        inp_dec_level2_image = self.reduce_chan_level2_image(inp_dec_level2_image)
        inp_dec_level2_event = self.reduce_chan_level2_event(inp_dec_level2_event)
        
        out_dec_level2_image, out_dec_level2_event = self.decoder_level2((inp_dec_level2_image, inp_dec_level2_event))

        inp_dec_level1_image = self.up2_1(out_dec_level2_image)
        inp_dec_level1_event = self.up2_1(out_dec_level2_event)
        
        inp_dec_level1_image = torch.cat([inp_dec_level1_image, out_enc_level1_image], 1)
        inp_dec_level1_event = torch.cat([inp_dec_level1_event, out_enc_level1_event], 1)
        
        out_dec_level1_image, out_dec_level1_event = self.decoder_level1((inp_dec_level1_image, inp_dec_level1_event))

        # Refinement & Global Residual Addition
        out_dec_level1_image, out_dec_level1_event = self.refinement((out_dec_level1_image, out_dec_level1_event))

        out_dec_level1_image = self.output_img(out_dec_level1_image) + img_inp
        out_dec_level1_event = self.output_eve(out_dec_level1_event)

        return out_dec_level1_image, out_dec_level1_event


def count_parameters(model):
    """Calculates the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)