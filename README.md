# Explainability and Robustness in Metric Space

This repository implements explainability and robustness methods such as Saliency Maps, Integrated Gradients, FGSM and PGD Attacks applied to point cloud classification models.

## Installation

- Set up and activate conda environment.

```bash
conda env create -f environment.yml
conda activate erms
```

- Install pre-commit hooks.

```bash
pre-commit install
```

- Download pretrained checkpoint.

```bash
mkdir out
gdown --no-cookies --id 1-2UrF5V_gpjGNbWfbk_kEp742Bnjlnc- -O out/pointmlp.pth
```

## Quick Start

- Run all experiments and plot results.

```bash
python run.py
```
