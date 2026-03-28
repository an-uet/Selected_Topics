from collections import Counter

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler


def data_sampler_imbalance(dataset):
    labels = dataset.labels

    counts = Counter(labels)

    class_weights = {}

    for c in range(100):

        if counts[c] == 0:
            class_weights[c] = 0
        else:
            class_weights[c] = 1.0 / counts[c]

    sample_weights = [class_weights[l] for l in labels]

    sample_weights = torch.tensor(sample_weights, dtype=torch.float)

    sampler = WeightedRandomSampler(
        sample_weights,
        len(sample_weights),
        replacement=True
    )

    return sampler


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)

    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)

    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)

    rand_index = torch.randperm(x.size(0)).to(x.device)

    y_a = y
    y_b = y[rand_index]

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)

    x[:, :, bbx1:bbx2, bby1:bby2] = \
        x[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) /
            (x.size(-1) * x.size(-2))
    )

    return x, y_a, y_b, lam

def plot_curves(history,save_path):

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(history["train_loss"],label="train")
    plt.plot(history["val_loss"],label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history["train_acc"],label="train")
    plt.plot(history["val_acc"],label="val")
    plt.title("Accuracy")
    plt.legend()

    plt.savefig(save_path)
    plt.close()



