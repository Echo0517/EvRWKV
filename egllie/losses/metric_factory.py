from absl.logging import info
from torch import nn
import torch

from losses.image_loss import EglliePSNR, EgllieSSIM,EglliePSNR_star

import pyiqa



class EgllieNIQE(nn.Module):
    """NIQE (Natural Image Quality Evaluator) 指标计算"""
    def __init__(self):
        super(EgllieNIQE, self).__init__()
        self.niqe_metric = pyiqa.create_metric('niqe', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, batch):
        pred_imgs = batch.get('pred', None)
        if pred_imgs is None:
            return torch.tensor(0.0)
        pred_imgs = pred_imgs.clamp(0, 1)
        niqe_score = self.niqe_metric(pred_imgs)
        return niqe_score.mean()


class EgllieBRISQUE(nn.Module):
    """BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) 指标计算"""
    def __init__(self):
        super(EgllieBRISQUE, self).__init__()
        self.brisque_metric = pyiqa.create_metric('brisque', device='cuda' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, batch):
        pred_imgs = batch.get('pred', None)
        if pred_imgs is None:
            return torch.tensor(0.0)
        
        # 确保图像值在[0,1]范围内
        pred_imgs = pred_imgs.clamp(0, 1)
        
        # 计算BRISQUE分数
        brisque_score = self.brisque_metric(pred_imgs)
        return brisque_score.mean()


def get_single_metric(config):
    if config.NAME == "SSIM":
        return EgllieSSIM()
    elif config.NAME == "PSNR":
        return EglliePSNR()
    elif config.NAME == "PSNR_star":
        return EglliePSNR_star()
    elif config.NAME == "NIQE":
        return EgllieNIQE()
    elif config.NAME == "BRISQUE":
        return EgllieBRISQUE()
    else:
        raise ValueError(f"Unknown config: {config}")


class MixedMetric(nn.Module):
    def __init__(self, configs):
        super(MixedMetric, self).__init__()
        self.metric = []
        self.eval = []
        for config in configs:
            try:
                self.metric.append(config.NAME)
                self.eval.append(get_single_metric(config))
            except ImportError as e:
                print(f"Warning: Skipping metric {config.NAME} due to: {e}")
        info(f"Init Mixed Metric: {configs}")

    def forward(self, batch):
        r = []
        for m, e in zip(self.metric, self.eval):
            r.append((m, e(batch)))
        return r