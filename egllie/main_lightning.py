"""
Main Training and Evaluation Script for the EvRWKV Framework.
This script utilizes PyTorch Lightning to handle the training loop, validation, 
testing, and visualization for the event-guided low-light image enhancement task.
"""

import torch
import os
import json
import yaml
import numpy as np
from absl import app, flags, logging
from absl.logging import info
from easydict import EasyDict
from pudb import set_trace
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import SGD, Adam, AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import lightning.pytorch as L
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from lightning.pytorch.strategies import DDPStrategy

import sys
sys.path.append('..')
from datasets import get_dataset
from losses import get_loss, AverageMeter, get_metric
from models import get_model
import random

import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

# Add NIQE support
import pyiqa  # Requires prior installation: pip install pyiqa

# -----------------------------------------------------------------------------
# Command Line Arguments Configuration
# -----------------------------------------------------------------------------
FLAGS = flags.FLAGS

flags.DEFINE_string("yaml_file", None, "The config file path.")
flags.DEFINE_string("RESUME_PATH", None, "The checkpoint path for resuming training.")
flags.DEFINE_string("RESUME_TYPE", None, "The type of resume operation.")
flags.DEFINE_boolean("RESUME_SET_EPOCH", False, "Whether to set the epoch when resuming.")
flags.DEFINE_boolean("TEST_ONLY", False, "Switch to test-only mode without training.")
flags.DEFINE_boolean("PUDB", False, "Enable PUDB debugging.")
flags.DEFINE_boolean(f"VISUALIZE", False, "Enable output visualization.")
flags.DEFINE_integer(f"TRAIN_BATCH_SIZE", None, "Override the default train batch size.")
flags.DEFINE_integer(f"VAL_BATCH_SIZE", None, "Override the default validation batch size.")

# Default logging directory
FLAGS.log_dir = 'log_dir_sdsd_in_cross'

def init_config(yaml_path):
    """
    Initialize and parse the experiment configuration.
    Reads the given YAML file, updates it with command-line FLAGS, 
    and returns an EasyDict configuration object.
    
    Args:
        yaml_path (str): Path to the YAML configuration file.
        
    Returns:
        EasyDict: Dictionary containing all experimental parameters.
    """
    if yaml_path is None:
        raise ValueError("The yaml_file parameter must be provided.")
    
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 0. Configure logging system and save directories
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()
    
    config["SAVE_DIR"] = os.path.join(FLAGS.log_dir, '..',  'checkpoints_sdsd_in_cross_1')
    if not os.path.exists(config["SAVE_DIR"]):
        os.makedirs(config["SAVE_DIR"])
        
    # 1. Resume settings
    if FLAGS.RESUME_PATH:
        config["RESUME"]["PATH"] = FLAGS.RESUME_PATH
        config["RESUME"]["TYPE"] = FLAGS.RESUME_TYPE
        config["RESUME"]["SET_EPOCH"] = FLAGS.RESUME_SET_EPOCH
        
    # 3. Visualization switch
    config["VISUALIZE"] = FLAGS.VISUALIZE
    
    # 4. Update batch size dynamically
    if FLAGS.TRAIN_BATCH_SIZE:
        info(f"Update TRAIN_BATCH_SIZE to {FLAGS.TRAIN_BATCH_SIZE}")
        config["TRAIN_BATCH_SIZE"] = FLAGS.TRAIN_BATCH_SIZE
    if FLAGS.VAL_BATCH_SIZE:
        info(f"Update VAL_BATCH_SIZE to {FLAGS.VAL_BATCH_SIZE}")
        config["VAL_BATCH_SIZE"] = FLAGS.VAL_BATCH_SIZE
        
    # 5. Test mode switch
    if FLAGS.TEST_ONLY:
        config["TEST_ONLY"] = FLAGS.TEST_ONLY
        
    # 6. Debug mode switch
    if FLAGS.PUDB:
        set_trace()

    info(f"Launch Config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)

def rot_aug(batch):
    """
    Data Augmentation: Applies synchronized random 90-degree rotations 
    to all multi-modal inputs (low-light, normal-light, event data, etc.) 
    to increase data diversity while maintaining spatial alignment.
    """
    rot_times = np.random.randint(0, 4)
    batch['lowligt_image'] = torch.rot90(batch['lowligt_image'], k=rot_times, dims=[2, 3])
    batch['normalligt_image'] = torch.rot90(batch['normalligt_image'], k=rot_times, dims=[2, 3])
    batch['event_free'] = torch.rot90(batch['event_free'], k=rot_times, dims=[2, 3])
    batch['lowlight_image_blur'] = torch.rot90(batch['lowlight_image_blur'], k=rot_times, dims=[2, 3])
    batch['ill_list'] = [torch.rot90(img, k=rot_times, dims=[2, 3]) for img in batch['ill_list']]
    return batch

class EvRGBModel(L.LightningModule):
    """
    Core PyTorch Lightning module. Encapsulates the EvRWKV network architecture, 
    loss function computation, evaluation metrics, and optimization strategies.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize core network, loss functions, and evaluation metrics
        self.model = get_model(self.config.MODEL)
        self.criterion = get_loss(self.config.LOSS)
        self.metrics = get_metric(self.config.METRICS)
        self.test_outputs = []
        
        # Lazy initialization for the No-Reference Image Quality Metric (NIQE)
        self._niqe_metric = None
        
        # Relative path for saving test prediction images (GitHub friendly)
        self.save_dir = "./visualization_results"
        os.makedirs(self.save_dir, exist_ok=True)

    @property
    def niqe_metric(self):
        """Lazy load the NIQE metric to save GPU memory when not needed."""
        if self._niqe_metric is None:
            try:
                self._niqe_metric = pyiqa.create_metric(
                    'niqe', 
                    device=self.device
                )
            except Exception as e:
                logging.warning(f"Failed to initialize NIQE metric: {e}")
                self._niqe_metric = None
        return self._niqe_metric

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        Supports four decay strategies: cosine, exponential, step, and plateau.
        """
        optimizer = Adam(self.model.parameters(), lr=self.config.OPTIMIZER.LR, weight_decay=self.config.OPTIMIZER.weight_decay)
        lr_scheduler_name = self.config.OPTIMIZER.LR_SCHEDULER.lower()
        
        if lr_scheduler_name == "cosine":
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.OPTIMIZER.end_epoch, eta_min=1e-8)
        elif lr_scheduler_name == "exponential":
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=self.config.OPTIMIZER.get('gamma', 0.95))
        elif lr_scheduler_name == "step":
            scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.OPTIMIZER.get('step_size', 10), gamma=self.config.OPTIMIZER.get('gamma', 0.2))
        elif lr_scheduler_name == "plateau":
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.OPTIMIZER.get('factor', 0.1),
                patience=self.config.OPTIMIZER.get('patience', 10),
                min_lr=1e-8
            )
        else:
            raise ValueError(f"Unsupported LR_SCHEDULER: {self.config.OPTIMIZER.LR_SCHEDULER}")
        
        if lr_scheduler_name == "plateau":
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',  
                    'interval': 'epoch',   
                    'frequency': 1        
                }
            }
        else:
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',  
                    'frequency': 1        
                }
            }

    def compute_niqe(self, pred_imgs):
        """
        Calculate the NIQE (Natural Image Quality Evaluator) score.
        Args:
            pred_imgs: Predicted image tensor (B, C, H, W), range must be [0, 1].
        Returns:
            niqe_score: Batch average NIQE score.
        """
        if self.niqe_metric is None:
            return None
            
        with torch.no_grad():
            try:
                niqe_score = self.niqe_metric(pred_imgs).mean()
                return niqe_score
            except Exception as e:
                logging.warning(f"NIQE calculation failed: {e}")
                return None
        
    def normalize_for_niqe(self, images):
        """
        Maps image tensors to the [0, 1] range using quantile normalization.
        This method is robust to extreme values/outliers and meets NIQE input requirements.
        """
        images = images.float()
        low_quantile = torch.quantile(images, 0.01)
        high_quantile = torch.quantile(images, 0.99)
        normalized = (images - low_quantile) / (high_quantile - low_quantile)
        normalized = torch.clamp(normalized, 0.0, 1.0)
        
        return normalized

    def training_step(self, batch, batch_idx):
        """
        Defines the forward pass and loss computation for a single training step.
        Includes synchronized spatial data augmentation and logs metrics.
        """
        batch_size = batch['lowligt_image'].size(0)
        batch = rot_aug(batch)
        outputs = self.model(batch)

        losses, name_to_loss = self.criterion(outputs)
        name_to_measure = self.metrics(outputs)

        # Log total loss, sub-losses, and evaluation metrics
        self.log("train_total_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        
        # Note: Indexing here depends on the specific implementation in losses.py / metrics.py
        self.log("train_"+name_to_loss[0][0], name_to_loss[0][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_"+name_to_loss[1][0], name_to_loss[1][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_"+name_to_measure[0][0], name_to_measure[0][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_"+name_to_measure[1][0], name_to_measure[1][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_"+name_to_measure[2][0], name_to_measure[2][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # NIQE is not computed during training to save computational resources
        return losses

    def validation_step(self, batch, batch_idx):
        """
        Defines the inference logic on the validation set.
        Calculates standard losses, full-reference metrics, and the no-reference NIQE metric.
        """
        batch_size = batch['lowligt_image'].size(0)
        outputs = self.model(batch)

        losses, name_to_loss = self.criterion(outputs)
        name_to_measure = self.metrics(outputs)

        self.log("valid_total_loss", losses, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("valid_"+name_to_loss[0][0], name_to_loss[0][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("valid_"+name_to_loss[1][0], name_to_loss[1][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("valid_"+name_to_measure[0][0], name_to_measure[0][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("valid_"+name_to_measure[1][0], name_to_measure[1][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("valid_"+name_to_measure[2][0], name_to_measure[2][1], on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Compute NIQE during validation
        pred_imgs = outputs.get('pred', None)
        if pred_imgs is not None:
            normalized_imgs = self.normalize_for_niqe(pred_imgs)
            niqe_score = self.compute_niqe(normalized_imgs)
            if niqe_score is not None:
                self.log("valid_NIQE", niqe_score, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return losses

    # -------------------------- Testing Functions --------------------------
    def test_step(self, batch, batch_idx):
        """Execute test mode inference and trigger local image saving."""
        batch_size = batch['lowligt_image'].size(0)
        outputs = self.model(batch)
        # Save output visual results
        self.save_test_images(batch, outputs, batch_idx)
        return outputs

    def save_test_images(self, batch, outputs, batch_idx):
        """
        Saves the predicted images, low-light inputs, and ground truth images 
        from the current batch to the local `self.save_dir`.
        Clamps pixel values to [0,1] to prevent saving artifacts.
        """
        pred_imgs = outputs.get('pred', None)
        if pred_imgs is None:
            return

        # Extract images: Low-light input and Ground Truth
        low_img = batch['lowligt_image']  # (B, C, H, W)
        gt_img  = batch['normalligt_image']

        # Process and save each sample in the batch individually
        B = low_img.size(0)
        for i in range(B):
            # Extract single image and move to CPU
            pred_i = pred_imgs[i].detach().cpu()
            low_i  = low_img[i].detach().cpu()
            gt_i   = gt_img[i].detach().cpu()

            # Convert to PIL Image and clamp values
            pred_pil = TF.to_pil_image(pred_i.clamp(0,1))
            low_pil  = TF.to_pil_image(low_i.clamp(0,1))
            gt_pil   = TF.to_pil_image(gt_i.clamp(0,1))

            base_name = f"batch{batch_idx:03d}_{i}"
            pred_filename = os.path.join(self.save_dir, base_name + "_pred.png")
            low_filename  = os.path.join(self.save_dir, base_name + "_input.png")
            gt_filename   = os.path.join(self.save_dir, base_name + "_gt.png")

            pred_pil.save(pred_filename)
            low_pil.save(low_filename)
            gt_pil.save(gt_filename)

# ---------------------------------------------------------------------

def main(args):
    """
    Main execution flow. Handles random seed fixation for reproducibility, 
    dataset loading, Distributed Data Parallel (DDP) setup, and initializes 
    the PyTorch Lightning Trainer.
    """
    config = init_config(FLAGS.yaml_file)
    
    # Fix random seeds to ensure experimental rigor and reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.backends.cudnn.benchmark = True

    train_dataset, val_dataset = get_dataset(config.DATASET)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=config.JOBS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.VAL_BATCH_SIZE,
        shuffle=False,
        num_workers=config.JOBS,
        pin_memory=True,
        drop_last=True,
    )

    # Configure automatic model checkpointing based on validation SSIM
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.SAVE_DIR,
        filename='{epoch:04d}-{valid_SSIM:.3f}', 
        monitor='valid_SSIM',
        every_n_epochs=1,
        save_top_k=10,
        mode='max',
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Load existing weights to resume training or initialize a new model
    if config.get("RESUME", {}).get("PATH"):
        model = EvRGBModel.load_from_checkpoint(
            config.RESUME.PATH,
            config=config,
            strict=True
        )
        print(f"Successfully loaded checkpoint: {config.RESUME.PATH}")
    else:
        model = EvRGBModel(config)
    
    
    # Initialize the Trainer with multi-GPU DDP strategy
    trainer = L.Trainer(
        max_epochs=config.END_EPOCH,
        devices=config.DEVICES,
        callbacks=[checkpoint_callback, lr_monitor],
        num_sanity_val_steps=2,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_true'
    )

    # Execute test-only pipeline or full train-validation pipeline
    if config.TEST_ONLY:
        # Testing only (Inference + Visualization saving)
        trainer.test(model, dataloaders=val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    FLAGS.yaml_file = './options/sdsd_in.yaml'
    print(f'=========={torch.cuda.device_count()}===========')
    app.run(main)