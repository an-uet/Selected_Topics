# Selected Topics in Visual Recognition using Deep Learning — HW3

**Student ID:** 314540018  
**Name:** Le Thi An  

---

## Introduction

This repository contains the training and evaluation pipeline for the HW3 instance segmentation task. It includes dataset preparation, environment setup, model training, inference, and visualization utilities.

---

## Environment Setup

Create and install the environment using the following steps:

```bash
# Create and activate conda environment
conda create -n hw3-env python=3.8
conda activate hw3-env

# Install PyTorch
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install Detectron2
python -m pip install "git+https://github.com/facebookresearch/detectron2.git"

# Install additional dependencies
pip install opencv-python
conda install shapely
```

---

## Dataset Preparation

Dataset path:

```bash
data/nycu-hw3-data/
```

The dataset must be converted into COCO format before training.

Run:

```bash
python convert_to_coco.py
```

This script will:

- Convert the dataset into COCO-style annotations
- Split the dataset into training and validation sets

Generated files:

- `full.json`
- `train.json`
- `val.json`
- `test_image_name_to_ids.json`

---

## Project Structure

```bash
hw3/
│
├── configs/              # Model configuration files
├── data/                 # Dataset and COCO annotations
├── visualize.ipynb            # Visualization utilities
├── train_mpvit.py        # MPViT training script
├── train_rcnn.py         # Mask R-CNN training script
├── test_mpvit.py         # MPViT inference script
├── test_tta.py           # TTA inference script
├── test_rcnn.py          # Mask R-CNN inference script
└── README.md
```

---

## Model Configuration

The experiments are based on configurations from:

- Detectron2
- MPViT

Detailed configuration files are provided in the `configs/` directory.

---

## Training

### MPViT

```bash
python train_mpvit.py \
    --config-file configs/maskrcnn/mask_rcnn_mpvit_base_ms_3x.yaml \
    --num-gpus 4
```

### Mask R-CNN (ResNet Backbone)

```bash
python train_rcnn.py \
    --model mask_rcnn_X_101_32x8d_FPN_3x.yaml \
    --output_dir checkpoints/models/X101_FPN3x
```

---

## Evaluation

### MPViT

```bash
python test_mpvit.py \
    --output_dir <output_dir> \
    --trained_model <checkpoint_folder>
```

### Test-Time Augmentation (TTA)

```bash
python test_tta.py \
    --output_dir <output_dir> \
    --trained_model <checkpoint_folder>
```

### Mask R-CNN

```bash
python test_rcnn.py \
    --output_dir <output_dir> \
    --trained_model <checkpoint_folder>
```

---

## Logging and Monitoring

- Training logs are stored using TensorBoard within each experiment directory
- Model checkpoints and best-performing weights are automatically saved in the output folders

---

## Performance Snapshot

![Demo](anlt.png)