<div align="center">

# EvRWKV: A Continuous Interactive RWKV Framework for Effective Event-Guided Low-Light Image Enhancement

[**IEEE TCSVT 2026**]

<div>
    <a href="#">
        <img src="https://doi.org/10.1109/TCSVT.2026.3672491" alt="Paper">
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


## :file_folder: Dataset Preparation

### Download Datasets

**SDE:**
- Baidu Pan: 
- Google Drive: 

**SDSD:**
- Baidu Pan: 
- Google Drive: 

**RELED:**
- Baidu Pan: 
- Google Drive: 

**LIE:**
- Baidu Pan: 
- Google Drive: 
---

## :computer: Usage

### 1. Dependencies & Installation

Clone the repository and set up the Conda environment:
```bash
cd EvRWKV

conda create -n evrwkv python=3.9
conda activate evrwkv

# Install PyTorch (CUDA 11.7 or higher is recommended)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install project dependencies
pip install -r requirements.txt
```

### 2. Pretrained Models

Checkpoints:

Baidu Pan: 
Google Drive: 

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
  author={Cai, Wenjie and Meng, Qingguo and Wang, Zhenyu and Dong, Xingbo and Jin, Zhe},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2026},
  publisher={IEEE}
}
```

## :heart: Acknowledgment

We thank the authors of [Restore-RWKV](https://github.com/Yaziwel/Restore-RWKV) and [EvLight](https://github.com/EthanLiang99/EvLight) for their open-source contributions.


## :email: Contact
If you have any questions, feel free to open an issue or contact:
**Wenjie Cai**: [wa2214030@stu.ahu.edu.cn](mailto:wa2214030@stu.ahu.edu.cn)