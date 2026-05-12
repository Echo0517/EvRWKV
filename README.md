<div align="center">

# EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement

[**IEEE TCSVT 2026**]

<div>
    <a href="#">
        <img src="https://img.shields.io/badge/Paper-TCSVT_2026-blue?style=flat-square" alt="Paper">
    </a>
    <a href="https://opensource.org/licenses/MIT" target="_blank">
        <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
    </a>
    <a href="https://pytorch.org/" target="_blank">
        <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square" alt="PyTorch">
    </a>
</div>

<br>

</div>

## :loudspeaker: News

- **[2026.05]** :tada: Our paper **"EvRWKV"** has been officially accepted by **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**!
- **[2026.05]** 🚀 The core training code, testing pipeline, custom CUDA operators, and network architecture are now open-source.

---

## :bulb: Introduction

**EvRWKV** proposes a novel event-guided low-light image enhancement framework. By leveraging the linear complexity and powerful spatial modeling capabilities of the RWKV architecture, we design a "continuous interactive" mechanism. This approach deeply fuses the static details of RGB images with the high dynamic range (HDR) and high temporal resolution characteristics of event cameras, achieving efficient and high-quality restoration of low-light scenes.

### Core Features:
- **Continuous Interactive Mechanism**: Achieves dynamic alignment and complementation of cross-modal features through a carefully designed Spatial Mix layer.
- **OmniShift Reparameterization**: Utilizes a multi-branch structure during training to capture multi-scale features, which collapses into a single-layer convolution during inference for optimal speed.
- **Custom CUDA Operator**: Features low-level optimization for WKV computations, supporting large-scale parallel processing of high-resolution images.

---

## :file_folder: Dataset Preparation

Please organize your dataset according to the following structure (using the SDSD dataset as an example). Before training, remember to update the `DATASET_ROOT` path in your YAML configuration files under the `egllie/options/` folder.

<details>
<summary><b>Click to view SDSD Directory Structure</b></summary>

```text
--indoor/outdoor 
├── test 
│   └── pair1 
│       ├── low 
│       │   ├── xxx.png (low-light RGB frame) 
│       │   ├── xxx.npz (split low-light event streams) 
│       │   └── lowligt_event.npz (the whole low-light event stream) 
│       └── normal 
│           └── xxx.png (normal-light RGB frame) 
└── train 
    └── pair1 
        ├── low 
        │   ├── xxx.png (low-light RGB frame) 
        │   ├── xxx.npz (split low-light event streams) 
        │   └── lowligt_event.npz (the whole low-light event stream) 
        └── normal 
            └── xxx.png (normal-light RGB frame) 
```
</details>

---

## :computer: Usage

### 1. Dependencies & Installation

Clone the repository and set up the Conda environment:
```bash
git clone [https://github.com/](https://github.com/)[Your_GitHub_Username]/EvRWKV.git
cd EvRWKV

conda create -n evrwkv python=3.9
conda activate evrwkv

# Install PyTorch (CUDA 11.7 or higher is recommended)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt
```

> :warning: **CUDA Operator Instructions**: 
> This project uses a custom WKV CUDA operator located in the `egllie/models/cuda/` directory. The code will perform just-in-time (JIT) compilation using `torch.utils.cpp_extension.load` during runtime. Please ensure that `nvcc` is installed and properly configured in your system's environment variables.

### 2. Pretrained Models

| Dataset | Metrics (PSNR / SSIM / NIQE) | Baidu Netdisk | OneDrive |
| :--- | :---: | :---: | :---: |
| **SDSD Indoor** | --.-- / --.-- / --.-- | [Link](#) (pwd: `xxxx`) | [Link](#) |
| **SDSD Outdoor**| --.-- / --.-- / --.-- | [Link](#) (pwd: `xxxx`) | [Link](#) |

*(Note: Pretrained weights will be uploaded soon.)*

### 3. Training
Navigate to the core directory and run the training script:
```bash
cd egllie
python main_lightning.py --yaml_file options/sdsd_in.yaml
```
> Training logs and checkpoints are saved in the `checkpoints_*/` directory in the parent folder by default.

### 4. Testing & Visualization
To load a pre-trained model and save the enhancement results:
```bash
cd egllie
python main_lightning.py \
  --yaml_file options/sdsd_in.yaml \
  --TEST_ONLY True \
  --RESUME_PATH /path/to/your/model.ckpt
```
> The visual inference results will be saved to `egllie/visualization_results/`.

---

## :mortar_board: Citation

If this work is helpful for your research, please consider citing:

```bibtex
@article{cai2026evrwkv,
  title={EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement},
  author={Cai, Wenjie and others},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## :heart: Acknowledgment

This project is built upon the foundational concepts of the [RWKV](https://github.com/BlinkDL/RWKV-LM) architecture. We extend our sincere gratitude to the open-source community and the authors of related works for their valuable contributions.

## :email: Contact
If you have any questions, feel free to open an issue or contact:
**Wenjie Cai**: [wa2214030@stu.ahu.edu.cn](mailto:wa2214030@stu.ahu.edu.cn)