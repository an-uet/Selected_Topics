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
