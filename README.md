# EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement

<p align="left">
  <a href="#"><img alt="Citation" src="https://img.shields.io/badge/Citation-TCSVT%202026-blue"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg"></a>
</p>

这是发表于 IEEE Transactions on Circuits and Systems for Video Technology (TCSVT) 2026 的论文 **"EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement"** 的官方 PyTorch 实现代码。

## 📢 最新动态 (News)

**[2026-05]** 🔥 我们的论文被 **TCSVT 2026** 正式录用！
**[2026-05]** 🚀 核心训练代码、测试管道及网络架构正式开源。

---

## 💡 简介 (Introduction)

**EvRWKV** 提出了一种基于事件引导（Event-Guided）的低光照图像增强框架。该框架利用了 RWKV 架构的线性复杂度和强大的空间建模能力，通过“连续交互”机制，将 RGB 图像的细节信息与事件相机的高动态范围（HDR）及高时间分辨率特性进行深度融合，实现了高效且高质量的低光照场景恢复。

**核心特性：**

**连续交互机制**：通过精心设计的空间混合（Spatial Mix）层，实现跨模态特征的动态对齐与补全。

**OmniShift 重参数化**：在训练阶段使用多分支结构捕获多尺度特征，推理阶段则等效转换为单层卷积，兼顾性能与效率。

**自定义 CUDA 算子**：针对 WKV 计算进行了底层优化，支持高分辨率图像的大规模并行处理。

---

## 🛠️ 安装与环境配置 (Installation)

**1. 克隆仓库**
```bash
git clone [https://github.com/](https://github.com/)[您的GitHub用户名]/EvRWKV.git
cd EvRWKV
```

**2. 创建虚拟环境**
```bash
conda create -n evrwkv python=3.9
conda activate evrwkv

# 安装 PyTorch (建议 CUDA 11.7 及以上)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装项目依赖项
pip install -r requirements.txt
```

**3. CUDA 算子说明**
本项目使用了自定义的 WKV CUDA 算子，位于 `egllie/models/cuda/` 目录下。代码在运行时会通过 `torch.utils.cpp_extension.load` 进行即时编译（JIT）。请确保您的系统中已安装 `nvcc` 且路径已配置到环境变量中。

---

## 📂 数据集准备 (Data Preparation)

请按照以下结构组织您的数据集（以 SDSD 数据集为例）：

```
--indoor/outdoor 
| 
----test 
|   | 
|   ----pair1 
|       | 
|       ----low 
|       |   | 
|       |   ----xxx.png (low-light RGB frame) 
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
在训练前，请在 `egllie/options/` 文件夹下的 YAML 配置文件中更新 `DATASET_ROOT` 路径。

---

## 🚀 训练与测试 (Usage)

所有运行脚本均位于 `egllie` 目录下。

**1. 训练模型 (Training)**
```bash
cd egllie
python main_lightning.py --yaml_file options/sdsd_in.yaml
```
训练日志和 Checkpoints 默认保存至上级目录的 `checkpoints_*/` 路径下。

**2. 测试与可视化 (Testing)**
如果您需要加载预训练模型并保存可视化结果：
```bash
cd egllie
python main_lightning.py \
  --yaml_file options/sdsd_in.yaml \
  --TEST_ONLY True \
  --RESUME_PATH /path/to/your/model.ckpt
```
推理结果将保存至 `egllie/visualization_results/`。

---

## ✒️ 引用 (Citation)

如果您在学术研究中使用了本代码或相关思想，请考虑引用我们的工作：

```bibtex
@article{cai2026evrwkv,
  title={EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement},
  author={Cai, Wenjie and others},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## 📄 开源协议 (License)
本项目代码遵循 [MIT License](LICENSE) 协议。

---
**联系方式**: [] / Wenjie Cai