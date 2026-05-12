# EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement

<p align="left">
  <a href="#"><img alt="Citation" src="https://img.shields.io/badge/Citation-TCSVT%202026-blue"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg"></a>
</p>

This is the official PyTorch implementation for the paper **"EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement"**, accepted by IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2026.

## 📢 News

**[2026-05]** 🔥 Our paper has been officially accepted by **TCSVT 2026**!
**[2026-05]** 🚀 The core training code, testing pipeline, and network architecture are now open-source.

---

## 💡 Introduction

**EvRWKV** proposes an event-guided low-light image enhancement framework. This framework leverages the linear complexity and powerful spatial modeling capabilities of the RWKV architecture. Through a "continuous interactive" mechanism, it deeply fuses the detailed information of RGB images with the high dynamic range (HDR) and high temporal resolution characteristics of event cameras, achieving efficient and high-quality restoration of low-light scenes.

**Core Features:**

* **Continuous Interactive Mechanism**: Achieves dynamic alignment and complementation of cross-modal features through a carefully designed Spatial Mix layer.
* **OmniShift Reparameterization**: Utilizes a multi-branch structure during the training phase to capture multi-scale features, which is equivalently converted into a single-layer convolution during inference, balancing performance and efficiency.
* **Custom CUDA Operator**: Features low-level optimization for WKV computations, supporting large-scale parallel processing of high-resolution images.

---

## 🛠️ Installation

**1. Clone the repository**
```bash
git clone [https://github.com/](https://github.com/)[Your_GitHub_Username]/EvRWKV.git
cd EvRWKV
```

**2. Create a virtual environment**
```bash
conda create -n evrwkv python=3.9
conda activate evrwkv

# Install PyTorch (CUDA 11.7 or higher is recommended)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt
```

**3. CUDA Operator Instructions**
This project uses a custom WKV CUDA operator located in the `egllie/models/cuda/` directory. The code will perform just-in-time (JIT) compilation using `torch.utils.cpp_extension.load` during runtime. Please ensure that `nvcc` is installed on your system and its path is configured in your environment variables.

---

## 📂 Data Preparation

Please organize your dataset according to the following structure (using the SDSD dataset as an example):

```text
--indoor/outdoor 
| 
----test 
|   | 
|   ----pair1 
|       | 
|       ----low 
|       |   | 
|       ----xxx.png (low-light RGB frame) 
|       |   ----xxx.npz (split low-light event streams) 
|       |   ----lowligt_event.npz (the whole low-light event stream) 
|       | 
|       ----normal 
|           | 
|           ----xxx.png (normal-light RGB frame) 
| 
----train 
    | 
    ----pair1 
        | 
        ----low 
        |   | 
        |   ----xxx.png (low-light RGB frame) 
        |   ----xxx.npz (split low-light event streams) 
        |   ----lowligt_event.npz (the whole low-light event stream) 
        | 
        ----normal 
            | 
            ----xxx.png (normal-light RGB frame) 
```

Before training, please update the `DATASET_ROOT` path in the YAML configuration files under the `egllie/options/` folder.

---

## 🚀 Usage

All execution scripts are located in the `egllie` directory.

**1. Training**
```bash
cd egllie
python main_lightning.py --yaml_file options/sdsd_in.yaml
```
Training logs and checkpoints are saved in the `checkpoints_*/` directory in the parent folder by default.

**2. Testing and Visualization**
If you need to load a pre-trained model and save the visualization results:
```bash
cd egllie
python main_lightning.py \
  --yaml_file options/sdsd_in.yaml \
  --TEST_ONLY True \
  --RESUME_PATH /path/to/your/model.ckpt
```
The inference results will be saved to `egllie/visualization_results/`.

---

## ✒️ Citation

If you use this code or our ideas in your academic research, please consider citing our work:

```bibtex
@article{cai2026evrwkv,
  title={EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement},
  author={Wenjie Cai, Qingguo Meng, Zhenyu Wang, Xingbo Dong, Zhe Jin},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## 📄 License
This project is open-sourced under the [MIT License](LICENSE).

---
**Contact**: [wa2214030@stu.ahu.edu.cn](mailto:wa2214030@stu.ahu.edu.cn) / Wenjie Cai