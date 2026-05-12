import numpy as np
import os
from torch.utils.data import Dataset, ConcatDataset
import cv2
import torch
import torch.nn.functional as F

class LIE_Dataset(Dataset):
    def __init__(
        self,
        dataset_root,
        height,
        width,
        seq_name,
        is_train,
        voxel_grid_channel,
    ):
        self.H = height
        self.W = width
        self.normal_img_folder = os.path.join(dataset_root, seq_name, "gt")
        self.low_img_folder = os.path.join(dataset_root, seq_name, "image")
        self.event_folder = os.path.join(dataset_root, seq_name, "event")

        # 获取排序后的文件列表
        self.low_img_list = sorted([f for f in os.listdir(self.low_img_folder) if f.endswith('.png')])
        self.normal_img_list = sorted([f for f in os.listdir(self.normal_img_folder) if f.endswith('.png')])
        self.event_list = sorted([f for f in os.listdir(self.event_folder) if f.endswith('.txt')])

        # 验证文件一致性
        len_low = len(self.low_img_list)
        len_normal = len(self.normal_img_list)
        len_event = len(self.event_list)
        
        assert len_low == len_normal == len_event, \
            f"文件数量不匹配\n" \
            f"低光图像: {len_low} ({self.low_img_folder})\n" \
            f"正常光图像: {len_normal} ({self.normal_img_folder})\n" \
            f"事件文件: {len_event} ({self.event_folder})"

        self.crop_size = (256, 256)  # 中心裁剪尺寸
        self.is_train = is_train
        self.voxel_grid_channel = voxel_grid_channel
        self.seq_name = seq_name

    def __len__(self):
        return len(self.low_img_list)

    def _load_events(self, event_path):
        """从txt文件加载事件数据，处理第一行是图像尺寸的情况"""
        events = []
        with open(event_path, 'r') as f:
            lines = f.readlines()
            
            # 检查第一行是否是图像尺寸信息
            first_line = lines[0].strip().split()
            if len(first_line) == 2 and all(part.isdigit() for part in first_line):
                # 第一行是图像尺寸，跳过
                start_idx = 1
            else:
                # 第一行是事件数据
                start_idx = 0
            
            # 解析事件数据
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) == 4:
                    try:
                        t, x, y, p = parts
                        # 转换为数值类型
                        t = float(t)
                        x = int(float(x))  # 先转float再转int，处理可能的浮点坐标
                        y = int(float(y))
                        p = int(p)
                        events.append([t, x, y, p])
                    except ValueError:
                        print(f"警告: 无法解析事件行: {line}")
                        continue
        
        if len(events) == 0:
            # 如果事件文件为空，返回一个虚拟事件
            print(f"警告: 事件文件为空: {event_path}")
            return np.array([[0, 0, 0, 0]], dtype=np.float32)
        
        events = np.array(events, dtype=np.float32)
        return events

    def _generate_voxel_grid(self, events, width, height):
        """生成体素网格"""
        if len(events) == 0:
            return torch.zeros((self.voxel_grid_channel, height, width), dtype=torch.float32)
        
        event_tensor = torch.from_numpy(events)
        
        # 确保坐标在有效范围内
        ex = torch.clamp(event_tensor[:, 1].long(), 0, width - 1)
        ey = torch.clamp(event_tensor[:, 2].long(), 0, height - 1)
        
        # 归一化时间戳到 [0, voxel_grid_channel-1]
        event_start = event_tensor[0, 0]
        event_end = event_tensor[-1, 0]
        
        # 避免除零
        if event_end - event_start <= 0:
            ch = torch.zeros(len(event_tensor), dtype=torch.long)
        else:
            ch = (
                (event_tensor[:, 0] - event_start) / (event_end - event_start) * (self.voxel_grid_channel - 1)
            ).long()
            torch.clamp_(ch, 0, self.voxel_grid_channel - 1)
        
        # 处理极性: 0 -> -1, 1 -> 1
        ep = event_tensor[:, 3].float()
        ep = torch.where(ep == 0, -1.0, 1.0)

        # 创建体素网格
        voxel_grid = torch.zeros(
            (self.voxel_grid_channel, height, width), dtype=torch.float32
        )
        
        # 使用index_put_累积事件
        if len(ch) > 0:  # 确保有事件
            voxel_grid.index_put_((ch, ey, ex), ep, accumulate=True)
        
        return voxel_grid

    def _center_crop(self, img):
        """对图像进行中心裁剪到256x256"""
        h, w = img.shape[:2]
        
        # 计算裁剪起始点
        start_h = (h - self.crop_size[0]) // 2
        start_w = (w - self.crop_size[1]) // 2
        
        # 确保裁剪区域在图像范围内
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(h, start_h + self.crop_size[0])
        end_w = min(w, start_w + self.crop_size[1])
        
        # 执行裁剪
        cropped = img[start_h:end_h, start_w:end_w]
        
        # 如果裁剪后尺寸小于目标尺寸，进行填充
        if cropped.shape[0] < self.crop_size[0] or cropped.shape[1] < self.crop_size[1]:
            pad_h = max(0, self.crop_size[0] - cropped.shape[0])
            pad_w = max(0, self.crop_size[1] - cropped.shape[1])
            
            # 计算填充量（上下左右均等填充）
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # 执行填充
            if img.ndim == 3:
                cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                                mode='constant', constant_values=0)
            else:
                cropped = np.pad(cropped, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                                mode='constant', constant_values=0)
        
        return cropped

    def _center_crop_voxel(self, voxel_tensor):
        """对体素网格进行中心裁剪到256x256"""
        _, h, w = voxel_tensor.shape
        
        # 计算裁剪起始点
        start_h = (h - self.crop_size[0]) // 2
        start_w = (w - self.crop_size[1]) // 2
        
        # 确保裁剪区域在体素网格范围内
        start_h = max(0, start_h)
        start_w = max(0, start_w)
        end_h = min(h, start_h + self.crop_size[0])
        end_w = min(w, start_w + self.crop_size[1])
        
        # 执行裁剪
        cropped = voxel_tensor[:, start_h:end_h, start_w:end_w]
        
        # 如果裁剪后尺寸小于目标尺寸，进行填充
        if cropped.shape[1] < self.crop_size[0] or cropped.shape[2] < self.crop_size[1]:
            pad_h = max(0, self.crop_size[0] - cropped.shape[1])
            pad_w = max(0, self.crop_size[1] - cropped.shape[2])
            
            # 计算填充量（上下左右均等填充）
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # 执行填充
            cropped = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), 
                           mode='constant', value=0)
        
        return cropped

    def _crop(self, input_frame_list, voxel_tensor):
        """统一对图像和体素网格进行中心裁剪到256x256"""
        cropped_image_list = []
        for img in input_frame_list:
            cropped_img = self._center_crop(img)
            # 转换为torch张量并调整维度
            if cropped_img.ndim == 2:  # 单通道图像
                cropped_img = np.expand_dims(cropped_img, axis=-1)
            cropped_image_list.append(
                torch.from_numpy(cropped_img).permute(2, 0, 1).float()
            )
        
        # 对体素网格进行中心裁剪
        cropped_voxel = self._center_crop_voxel(voxel_tensor)
        
        return cropped_image_list, cropped_voxel

    def _illumination_map(self, img):
        """生成光照先验图（保持通道维度）"""
        return np.max(img, axis=2, keepdims=True) / 255.0

    def __getitem__(self, index):
        # 加载低光图像
        img_low = cv2.cvtColor(
            cv2.imread(os.path.join(self.low_img_folder, self.low_img_list[index])),
            cv2.COLOR_BGR2RGB,
        )
        
        # 加载正常光图像
        img_gt = cv2.cvtColor(
            cv2.imread(os.path.join(self.normal_img_folder, self.normal_img_list[index])),
            cv2.COLOR_BGR2RGB,
        )
        
        # 加载事件并生成体素网格
        event_path = os.path.join(self.event_folder, self.event_list[index])
        events = self._load_events(event_path)
        
        # 生成体素网格 (使用原始图像尺寸)
        voxel_tensor = self._generate_voxel_grid(events, self.W, self.H)

        # 预处理
        img_blur = cv2.blur(img_low, (5, 5))
        img_low_ill = self._illumination_map(img_low)

        # 中心裁剪到256x256
        cropped_imgs, cropped_voxel = self._crop(
            [img_low, img_gt, img_low_ill, img_blur],
            voxel_tensor
        )

        # 构建样本字典
        sample = {
            "lowligt_image": torch.pow(cropped_imgs[0] / 255.0, 0.45), 
            "normalligt_image": cropped_imgs[1] / 255.0,
            "event_free": cropped_voxel,
            "lowlight_image_blur": cropped_imgs[3] / 255.0,
            "ill_list": [cropped_imgs[2]],
            "seq_name": self.seq_name,
            "frame_id": self.low_img_list[index].split(".")[0],
        }
        return sample

def get_lie_dataset(
    dataset_root,
    is_train,
    voxel_grid_channel
):
    all_seqs = [d for d in os.listdir(dataset_root) 
                if os.path.isdir(os.path.join(dataset_root, d))]
    all_seqs.sort()

    seq_dataset_list = []
    for seq in all_seqs:
        seq_path = os.path.join(dataset_root, seq)
        
        # 自动获取图像尺寸
        sample_img_path = os.path.join(seq_path, "image", os.listdir(os.path.join(seq_path, "image"))[0])
        sample_img = cv2.imread(sample_img_path)
        H, W = sample_img.shape[:2]

        seq_dataset_list.append(
            LIE_Dataset(
                dataset_root=dataset_root,
                height=H,
                width=W,
                seq_name=seq,
                is_train=is_train,
                voxel_grid_channel=voxel_grid_channel
            )
        )
    return ConcatDataset(seq_dataset_list)