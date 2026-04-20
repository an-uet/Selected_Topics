# Selected Topics in Visual Recognition using Deep Learning — HW2

**Student ID:** 314540018  
**Name:** Le Thi An  

---

## Introduction

This repository contains the training pipeline for the HW2 object detection task. It includes dataset preparation, environment setup, and training scripts for two model families:

- **Deformable DETR** (`Deformable-DETR/`)
- **DETR** (`detr/`)

---

## Repository Structure

    .
    ├── Deformable-DETR/      # Deformable DETR implementation
    ├── detr/                 # DETR implementation
    ├── data/
    │   └── nycu-hw2-data/    # Dataset (COCO format)
    │       ├── train.json
    │       ├── valid.json
    │       └── test/

---

## Dataset

- Dataset path: `data/nycu-hw2-data/`  
- Annotation format: **COCO-style JSON**

Files:
- `train.json`
- `valid.json`
- Test images: `data/nycu-hw2-data/test/`

---

## Environment Setup

Please follow the official setup instructions:

- `Deformable-DETR/README.md`
- `detr/README.md`

---

## Key Parameters

For full configurations, refer to:

- `detr/main.py`
- `Deformable-DETR/main.py`

### Core Parameters

- `name`: Experiment name  
- `backbone`: Feature extractor (e.g., `resnet50`)  
- `data_dir`: Dataset root directory  
- `num_classes`: Number of object classes  

### Optimization

- `lr`: Learning rate for transformer and detection head (e.g., `1e-4`)  
- `lr_backbone`: Learning rate for backbone (e.g., `1e-5`)  
- `batch_size`: Batch size (e.g., `2`)  
- `epochs`: Number of training epochs (e.g., `300`)  
  - *Note: the model typically converges within 20–30 epochs*  
- `lr_drop`: Epoch to decay learning rate (e.g., `10`)  
- `weight_decay`: Weight decay  

### Model Architecture

- `enc_layers`: Number of encoder layers (e.g., `3`)  
- `dec_layers`: Number of decoder layers (e.g., `3`)  
- `hidden_dim`: Embedding dimension (e.g., `128`)  
- `dim_feedforward`: Feedforward dimension (e.g., `512`)  
- `nheads`: Number of attention heads (e.g., `4`)  
- `num_queries`: Number of object queries (e.g., `10`, default `100`)  

### Loss and Matching

- `bbox_loss_coef`: Weight for bounding box loss  
- `giou_loss_coef`: Weight for GIoU loss  
- `set_cost_bbox`: Matching cost for bounding box  
- `set_cost_giou`: Matching cost for GIoU  

### Checkpointing

- `output_dir`: Directory to save logs and checkpoints  
- `resume`: Path to resume checkpoint  

---

## Training Commands

### Deformable DETR

    bash Deformable-DETR/run.sh

---

### DETR

    bash detr/run.sh

---

## Evaluation / Inference

Use the provided scripts:

- `detr/inference.py`
- `Deformable-DETR/inference.py`

Example:

    python inference.py <checkpoint_path> <output_dir> <prediction_file_name> <cuda/cpu>

---

## Logging and Monitoring

- Training logs are saved in `log.txt` within each experiment folder  
- Checkpoints and best model weights are stored in output directories  

---

Performance Snapshot
-------
![Demo](demo.png)

## Acknowledgements

This project is based on code from the following public repositories:
- DETR: https://github.com/facebookresearch/detr
- Deformable DETR: https://github.com/fundamentalvision/Deformable-DETR

We made minor modifications to adapt the code for our specific task.
