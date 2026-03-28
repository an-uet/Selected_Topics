import os

import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pandas as pd
import timm
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ImageData
from utils import mixup_data, cutmix_data, data_sampler_imbalance, plot_curves


class TwoLayerHead(nn.Module):
    def __init__(self, in_features, num_classes=100, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.BatchNorm1d(in_features),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def attach_head(model, head):
    """
    Attach classification head to different model types
    """
    if hasattr(model, "fc"):  # torchvision ResNet
        model.fc = head
    elif hasattr(model, "classifier"):  # EfficientNet, ConvNeXt...
        model.classifier = head
    elif hasattr(model, "head"):  # ViT, some timm models
        model.head = head
    else:
        raise AttributeError("Unknown model head structure")
    return model


def build_model(model_name, head_type="1_FC", num_classes=100, pretrained=True):
    if model_name in ["resnet18","resnet34", "resnet50", "resnet101"]:
        # torchvision models
        model = getattr(torchvision.models, model_name)(
            weights="IMAGENET1K_V1" if pretrained else None
        )
        in_features = model.fc.in_features

    else:
        # timm models
        if head_type == "1_FC":
            # directly use built-in classifier
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes
            )
            return model

        else:
            # remove classifier
            model = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0
            )
            in_features = model.num_features

    # build head
    if head_type == "1_FC":
        head = nn.Linear(in_features, num_classes)

    elif head_type == "2_FC":
        head = TwoLayerHead(in_features, num_classes)

    else:
        raise ValueError(f"Unknown head_type: {head_type}")

    model = attach_head(model, head)

    return model


def train_epoch(model, loader, optimizer, criterion, aug=None, device='cuda'):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(loader):
        imgs = batch["img"].to(device)
        labels = batch["target"].to(device)

        if aug == "mixup":
            imgs, y_a, y_b, lam = mixup_data(imgs, labels)
            preds = model(imgs)
            loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

        elif aug == "cutmix":
            imgs, y_a, y_b, lam = cutmix_data(imgs, labels)
            preds = model(imgs)
            loss = lam * criterion(preds, y_a) + (1 - lam) * criterion(preds, y_b)

        else:
            preds = model(imgs)
            loss = criterion(preds, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = preds.argmax(1)

        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device='cuda', return_probs=False):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["img_w"].to(device)
            labels = batch["target"].to(device)

            preds = model(imgs)  # logits
            loss = criterion(preds, labels)
            total_loss += loss.item()

            # ---- convert to probabilities ----
            probs = F.softmax(preds, dim=1)

            pred = probs.argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

            if return_probs:
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

    if return_probs:
        all_probs = torch.cat(all_probs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        return total_loss / len(loader), correct / total, all_probs, all_labels

    return total_loss / len(loader), correct / total


def generate_prediction(model, test_dir, transform, save_path, device='cuda'):
    model.eval()
    results = []

    for img_name in sorted(os.listdir(test_dir)):
        path = os.path.join(test_dir, img_name)

        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)
            pred = torch.argmax(out, 1).item()

        name = os.path.splitext(img_name)[0]
        results.append([name, pred])

    df = pd.DataFrame(results, columns=["image_name", "pred_label"])
    df.to_csv(save_path, index=False)
    print("Saved:", save_path)



def generate_prediction_v2(
    model,
    test_dir,
    transform,
    save_path,          # file pred_label
    prob_save_path,     # file probability
    device='cuda'
):
    model.eval()

    results = []        # original output
    prob_results = []   # full probabilities
    prob_list = []

    for img_name in sorted(os.listdir(test_dir)):
        path = os.path.join(test_dir, img_name)

        img = Image.open(path).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img)                  # logits
            probs = F.softmax(out, dim=1)     # probabilities
            pred = probs.argmax(1).item()

        name = os.path.splitext(img_name)[0]

        results.append([name, pred])

        prob_list = probs.squeeze(0).cpu().tolist()
        prob_results.append([name] + prob_list)

    df = pd.DataFrame(results, columns=["image_name", "pred_label"])
    df.to_csv(save_path, index=False)

    num_classes = len(prob_list)
    prob_columns = ["image_name"] + [f"prob_{i}" for i in range(num_classes)]
    df_prob = pd.DataFrame(prob_results, columns=prob_columns)
    df_prob.to_csv(prob_save_path, index=False)

    print(f"Saved prediction: {save_path}")
    print(f"Saved probabilities: {prob_save_path}")


def run(model_config, device='cuda'):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(288),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    val_tf = transforms.Compose([
        transforms.CenterCrop(288),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dir = f"{model_config['data_dir']}/train"
    test_dir = f"{model_config['data_dir']}/test"
    val_dir = f"{model_config['data_dir']}/val"

    train_dataset = ImageData(train_dir,
                              transform_w=train_tf,
                              pseudo_csv=model_config['pseudo_csv'] if model_config["use_pseudo"] else None,
                              pseudo_img_dir=test_dir if model_config["use_pseudo"] else None
                              )

    val_dataset = ImageData(val_dir, val_tf)

    sampler = data_sampler_imbalance(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=model_config["batch_size"],
        sampler=sampler,
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=model_config["batch_size"],
        shuffle=False,
        num_workers=4
    )


    print("\nRunning:", model_config["name"])

    model = build_model(model_config["model"], head_type=model_config['head_type']).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=model_config["label_smoothing"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(model_config["lr"]),
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=model_config["epochs"])
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": []}

    best_acc = 0

    # Setup TensorBoard SummaryWriter
    log_dir = f"outputs/{model_config['name']}/runs"
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(model_config["epochs"]):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, model_config["aug"])

        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"{model_config['name']} Epoch {epoch} "
              f"train_acc {train_acc:.4f} "
              f"val_acc {val_acc:.4f}")

        # Log scalars to TensorBoard
        writer.add_scalar('loss/train', train_loss, epoch)
        writer.add_scalar('loss/val', val_loss, epoch)
        writer.add_scalar('acc/train', train_acc, epoch)
        writer.add_scalar('acc/val', val_acc, epoch)
        # log learning rate
        try:
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('lr', current_lr, epoch)
        except Exception:
            pass

        if val_acc > best_acc:
            best_acc = val_acc

            torch.save(model.state_dict(), f"outputs/{model_config['name']}_best.pth")

        plot_curves(
            history,
            f"outputs/{model_config['name']}_curves.png"
        )

        generate_prediction(
            model,
            test_dir,
            val_tf,
            f"outputs/{model_config['name']}_prediction.csv"
        )

        os.makedirs(f"outputs/{model_config['name']}", exist_ok=True)
        generate_prediction(
            model,
            test_dir,
            val_tf,
            f"outputs/{model_config['name']}/prediction.csv.zip"
        )

    # Close the TensorBoard writer
    writer.close()

if __name__ == '__main__':

    with open("model_config.yaml", "r") as f:
        model_configs = yaml.safe_load(f)

    run(model_configs)
