Introduction
------------
This repository contains a training pipeline for image classification for the HW1. It includes:
- Dataset loader: `codes/dataset.py` (class ImageData)
- Training and evaluation: `codes/main.py`
- Utility functions for augmentation and plotting: `codes/utils.py`
- Example model configuration: `codes/model_config.yaml`

The code supports mixup and cutmix data augmentations, class-imbalance sampling, and exporting predictions and training curves to an `outputs/` folder.

Environment Setup
-----------------
Recommended: create and activate a virtual environment, then install dependencies.

1) Create and activate a conda environment (if using conda):

```bash
conda create -n image_classification python=3.10 -y
conda activate hw1_img_classification
```

2) Install required packages:

```bash
pip install --upgrade pip
pip install numpy pandas matplotlib pillow tqdm torch torchvision timm
```

Optional: pin versions in a `requirements.txt` for reproducibility.

Usage
-----
1) Prepare your dataset directory in the expected structure (used by `ImageData`):

```
<data_dir>/
    train/
        0/
            img1.jpg
            img2.jpg
        1/
            img3.jpg
            ...
    val/
        0/
        1/
    test/
        000001.jpg
        000002.jpg
        ...
```

2) Edit `codes/model_config.yaml` to point `data_dir` at your dataset and adjust hyperparameters (lr, epochs, batch_size, aug, etc.). Example config keys: `name`, `model`, `data_dir`, `lr`, `epochs`, `batch_size`, `aug`, `use_pseudo`, `pseudo_csv`.
Configuration
- `name`: The experiment name or run identifier
- 
- `model`: The architecture name to use. Examples: `resnet18`, `resnet34`, `resnet50`, `resnet101 `. You can also provide a `timm` model string if using timm models.

- `pretrained`: `True`/`False` — if `True`, load ImageNet-pretrained weights where available.

- `data_dir`: Path to the dataset root directory. The code expects subfolders `train/`, `val/`, and `test/` inside this directory

- `lr`: Initial learning rate for the optimizer (example: `1e-4`).

- `epochs`: Number of training epochs to run (example: `50`).

- `label_smoothing`: Label smoothing factor used when constructing the classification loss (CrossEntropyLoss). A small value (e.g. `0.1`) softens hard one-hot targets.

- `batch_size`: Batch size used by the DataLoader for both training and validation.

- `head_type`: Classification head type to attach to the backbone. `1_FC` means a single fully-connected output layer; `2_FC` uses a small two-layer head 

- `aug`: Augmentation strategy for training. Supported values in the script: `cutmix`, `mixup`, or leave blank/other for no special mixing augmentations.

- `use_pseudo`: `True`/`False` to enable pseudo-labeling — if `True` the script will expect pseudo labels to be provided and use them during training according to the code's pseudo-label logic.

- `pseudo_csv`: Path to a CSV file containing pseudo labels (or `null` if not using pseudo-labeling).

3) Run training / evaluation from the `codes/` directory (the script loads `model_config.yaml` from the current working directory):

```bash
cd codes
python3 main.py
```

TensorBoard
-----------
If you enabled TensorBoard logging (the code uses torch.utils.tensorboard.SummaryWriter), logs are written to:

```
outputs/<name>/runs/
```

To view them during or after training:

```bash
# from the repo root
tensorboard --logdir codes/outputs/<name>/runs --bind_all
```

Outputs
-------
During training the script writes files to `outputs/`:
- `outputs/<name>_best.pth` — best model weights (state_dict)
- `outputs/<name>_curves.png` — training/validation loss & accuracy plot
- `outputs/<name>_prediction.csv` — test set predictions
- `outputs/<name>/prediction.csv.zip` — (zipped) predictions in a namespaced folder
