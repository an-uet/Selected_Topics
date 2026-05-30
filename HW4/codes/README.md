# Selected Topics in Visual Recognition using Deep Learning — HW4

**Student ID:** 315450018
**Name:** Le Thi An

---

## Overview

This repository contains the implementation for **Image Restoration** in HW4 of the course **Selected Topics in Visual Recognition using Deep Learning**.

The project includes:

* Dataset preparation
* Environment setup
* Model training with PromptIR
* Evaluation and inference
* Visualization utilities

---

## Environment Setup

Create and activate the environment:

```bash
conda env create -f environment.yml
conda activate hw4-env
```

Install additional dependencies manually if needed:

```bash
pip install opencv-python
pip install tqdm
pip install tensorboard
pip install lightning
```

---

## Dataset Preparation

Dataset structure:

```bash
data/hw4_realse_dataset/
├── train/
│   ├── clean/
│   └── degraded/
└── test/
```

Prepare the dataset:

```bash
python data/preprocess.py
```

This script will:

* Generate text files listing rain/snow image paths
* Create train/validation splits

Generated files:

```bash
rain.txt
snow.txt
```

---

## Project Structure

```bash
HW4/
│
├── net/                       # PromptIR model implementation
│   └── model.py
├── utils/
│   ├── dataset_utils.py       # Dataset loading and augmentation
│   ├── degradation_utils.py   # Degradation transforms
│   ├── inference_utils.py     # Inference helpers: pad_input, tta_eval, tile_eval
│   ├── loss_utils.py          # Edge loss and frequency loss
│   ├── schedulers.py          # LR scheduler
│   └── val_utils.py           # PSNR / SSIM computation
├── data/                      # Dataset and preprocessing scripts
├── output/                    # Saved predictions
├── train_ckpt/                # Saved model checkpoints
│
├── train.py                   # Training script (defines PromptIRModel)
├── demo.py                    # Inference script
├── example_img2npz.py         # Convert output to .npz submission format
├── options.py                 # Training hyperparameters
│
└── README.md
```

---

## Model

### Architecture

The implementation is based on:

* **PromptIR** — an all-in-one blind image restoration model with prompt-based guidance
* **PyTorch Lightning** — training framework

Key components:

* PromptIR backbone with decoder (`net/model.py`)
* Edge-aware loss for structural detail preservation (`utils/loss_utils.py`)
* Frequency-domain loss for perceptual quality (`utils/loss_utils.py`)
* Linear warmup cosine annealing scheduler (`utils/schedulers.py`)

The combined training loss:

```
loss = L1 + 0.1 * edge_loss + 0.05 * freq_loss
```

---

## Training

Launch training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
    --epochs 1000 \
    --batch_size 4 \
    --lr 2e-4 \
    --patch_size 128 \
    --num_gpus 4
```

Example with augmentation:

```bash
python train.py \
    --num_aug 20 \
    --ckpt_dir train_ckpt_aug20
```

---

## Evaluation

Evaluate a trained checkpoint by running inference and computing PSNR/SSIM on the validation set:

```bash
python demo.py \
    --output_path output/val_results \
    --ckpt_name best_model.ckpt \
    --ckpt_dir train_ckpt/
```

---

## Inference

Run inference on test images:

```bash
python demo.py \
    --output_path output/rainsnow_edge_hf \
    --ckpt_name best_rainsnow_edge-epoch=153-psnr=35.85.ckpt \
    --ckpt_dir train_ckpt_freq/
```

Convert predictions to `.npz` submission format:

```bash
python example_img2npz.py
```

---

## Logging

Training statistics are logged using:

* TensorBoard
* Console logs

Launch TensorBoard:

```bash
tensorboard --logdir logs
```

---


### Performance Snapshot

![Demo](anlt.png)

