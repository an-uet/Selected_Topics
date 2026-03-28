import os

import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class ImageData(Dataset):

    def __init__(
            self,
            root_dir,
            transform_w=None,
            pseudo_csv=None,
            pseudo_img_dir=None
    ):

        self.root_dir = root_dir
        self.transform_w = transform_w

        self.image_paths = []
        self.labels = []

        for label_str in sorted(os.listdir(root_dir)):
            label_dir = os.path.join(root_dir, label_str)
            if os.path.isdir(label_dir):
                label = int(label_str)
                for file_name in os.listdir(label_dir):

                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        path = os.path.join(label_dir, file_name)
                        self.image_paths.append(path)
                        self.labels.append(label)

        print(f"Original data: {len(self.image_paths)}")

        # Load pseudo labels
        if pseudo_csv is not None:
            df = pd.read_csv(pseudo_csv)
            for _, row in df.iterrows():
                img_name = str(row["image_name"])
                label = int(row["pred_label"])
                img_path = os.path.join(pseudo_img_dir, img_name + ".jpg")
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(label)

            print(f"After pseudo-label: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(path).convert("RGB")

        if self.transform_w:
            img = self.transform_w(img)

        return {
            "img": img,
            "target": label}
