# CIDNet: Collaborative Uncertainty-Aware Learning of Invariance and Discrimination for Cross-View Geo-Localization

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition-blue)](https://www.sciencedirect.com/science/article/abs/pii/S0031320326021230)
**Official implementation** of the paper **"CIDNet: Collaborative Uncertainty-Aware Learning of Invariance and Discrimination for Cross-View Geo-Localization"**, published in **Pattern Recognition (2026)**.

**Authors:** Jianghua Li, Yixin He, Yabo Hui, Tengyu Liu, and Xiaofeng Wang.

This repository provides the code and instructions for training and evaluating CIDNet on cross-view geo-localization benchmarks. Pretrained model checkpoints will be shared via Google Drive.

## News

- **October 2026:** Our paper, **CIDNet**, was published in *Pattern Recognition*.
- **Pretrained checkpoints:** Google Drive download links will be added after the model weights are uploaded.

## Overview

Cross-view geo-localization aims to identify the geographical location of a query image by matching it against reference images captured from different viewpoints, such as drone and satellite views. Large viewpoint changes make it challenging to learn representations that are both robust to cross-view variations and sufficiently discriminative for accurate retrieval.

**CIDNet** investigates **collaborative uncertainty-aware learning of invariance and discrimination** for cross-view geo-localization. The method is designed to jointly consider view-invariant information and discriminative information while accounting for uncertainty during learning.

This repository includes instructions for the following benchmarks:

- **University-1652**
- **SUES-200**

For the full methodology, experimental settings, and quantitative comparisons, please refer to the paper.

## Installation

Clone this repository and create a Python environment for CIDNet.

```bash
conda create -n cidnet python=3.10 -y
conda activate cidnet
```

Install PyTorch and the other dependencies required by the released code. If the repository includes a `requirements.txt` file, you can install the listed dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** Python 3.10 is a suggested starting environment, not a verified requirement. Please match the Python, PyTorch, CUDA, and package versions to the released implementation and your hardware.

## Dataset Preparation

Download the datasets from their official project pages:

- **University-1652:** [University-1652 Baseline](https://github.com/layumi/University1652-Baseline)
- **SUES-200:** [SUES-200 Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark)

Extract the datasets and preserve the directory layouts shown below. Make sure their root paths are consistent with the paths expected by the dataset loader in the code.

### University-1652

```text
University-1652/
├── train/
│   ├── drone/
│   │   ├── 0001/
│   │   ├── 0002/
│   │   └── ...
│   └── satellite/
└── test/
    ├── query_drone/
    ├── gallery_drone/
    ├── query_satellite/
    └── gallery_satellite/
```

### SUES-200

```text
SUES-200/
├── Training/
│   ├── 150/
│   ├── 200/
│   ├── 250/
│   └── 300/
└── Testing/
    ├── 150/
    ├── 200/
    ├── 250/
    └── 300/
```

## Training

The main entry point is `main.py`. Set `--only_test=False` to train a model.

**University-1652**

```bash
python main.py --only_test=False --dataset_name='University-1652'
```

**SUES-200**

```bash
python main.py --only_test=False --dataset_name='SUES-200'
```

Before running training, ensure that the dataset paths and other training settings in the released code are configured for your environment.

## Evaluation

Set `--only_test=True` and specify the checkpoint file using `--ckpt_path`.

**University-1652**

```bash
python main.py --only_test=True --dataset_name='University-1652' --ckpt_path='/path/to/university1652_checkpoint.pth'
```

**SUES-200**

```bash
python main.py --only_test=True --dataset_name='SUES-200' --ckpt_path='/path/to/sues200_checkpoint.pth'
```

Replace the example checkpoint paths with the actual paths to your downloaded or locally trained weights.

> The command-line argument names above follow the original repository instructions. Please ensure that the published `main.py` uses the same argument format.

## Pretrained Checkpoints

Pretrained CIDNet model weights will be provided through **Google Drive**.

| Dataset         | Checkpoint                     |
| :-------------- | :----------------------------- |
| University-1652 | **Coming soon** — Google Drive |
| SUES-200        | **Coming soon** — Google Drive |

Once the checkpoints are available, download the appropriate file and pass its local path to `--ckpt_path` for evaluation.

## Results

Please refer to the **CIDNet paper in *Pattern Recognition*** for the reported benchmark results, evaluation protocols, and comparisons with existing methods.

The results may depend on the dataset split, training configuration, random seed, and software environment. For reproducibility, use the experimental settings described in the paper.

## Citation

If you find CIDNet useful in your research, please consider citing our paper:

```bibtex
@article{li2026cidnet,
  title   = {CIDNet: Collaborative Uncertainty-Aware Learning of Invariance and Discrimination for Cross-View Geo-Localization},
  author  = {Li, Jianghua and He, Yixin and Hui, Yabo and Liu, Tengyu and Wang, Xiaofeng},
  journal = {Pattern Recognition},
  year    = {2026},
  publisher = {Elsevier}
}
```

*The citation above includes the currently verified publication information. Volume, article number, and DOI can be added once the final publisher metadata is confirmed.*

## Acknowledgements

We gratefully acknowledge the following open-source projects, which were referenced in the development of this repository:

- [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
- [MCCG](https://github.com/mode-str/crossview)
- [DAC](https://github.com/SummerpanKing/DAC)

We also thank the creators of **University-1652** and **SUES-200** for making their benchmarks available to the research community.
