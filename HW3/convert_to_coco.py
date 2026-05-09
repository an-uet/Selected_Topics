import os
import cv2
import json
import glob
import numpy as np
import skimage.io as sio

from pathlib import Path
from sklearn.model_selection import train_test_split


# ============================================================
# Geometry Utilities
# ============================================================

class PolygonExtractor:

    @staticmethod
    def binary_to_segments(binary_mask):

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        segments = []

        for contour in contours:

            coords = contour.reshape(-1, 2)

            if coords.shape[0] < 3:
                continue

            segments.append(coords.flatten().tolist())

        return segments

    @staticmethod
    def polygon_to_bbox(segment):

        pts = np.asarray(segment).reshape(-1, 2)

        xmin = pts[:, 0].min()
        ymin = pts[:, 1].min()

        xmax = pts[:, 0].max()
        ymax = pts[:, 1].max()

        return [
            float(xmin),
            float(ymin),
            float(xmax - xmin),
            float(ymax - ymin)
        ]

    @staticmethod
    def polygon_area(segment):

        contour = np.asarray(segment).reshape(-1, 2)

        return float(
            cv2.contourArea(contour)
        )


# ============================================================
# Dataset Builder
# ============================================================

class CocoAnnotationBuilder:

    def __init__(self):

        self.payload = {
            "images": [],
            "annotations": [],
            "categories": []
        }

        self.image_counter = 0
        self.annotation_counter = 0

        self.discovered_categories = set()

    def register_image(self, path, shape):

        h, w = shape[:2]

        image_info = {
            "file_name": path,
            "height": h,
            "width": w,
            "id": self.image_counter
        }

        self.payload["images"].append(image_info)

        current_id = self.image_counter

        self.image_counter += 1

        return current_id

    def register_annotation(
        self,
        image_id,
        category_id,
        polygon
    ):

        bbox = PolygonExtractor.polygon_to_bbox(polygon)

        area = PolygonExtractor.polygon_area(polygon)

        record = {
            "segmentation": [polygon],
            "iscrowd": 0,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "area": area,
            "id": self.annotation_counter
        }

        self.payload["annotations"].append(record)

        self.annotation_counter += 1

    def finalize_categories(self):

        for cid in sorted(self.discovered_categories):

            self.payload["categories"].append({
                "id": cid,
                "name": f"class{cid}",
                "supercategory": "none"
            })

    def export(self, save_path):

        self.finalize_categories()

        with open(save_path, "w") as fp:
            json.dump(
                self.payload,
                fp,
                indent=2
            )


# ============================================================
# Dataset Parser
# ============================================================

class InstanceDatasetParser:

    def __init__(self, root_dir):

        self.root_dir = Path(root_dir)

        self.builder = CocoAnnotationBuilder()

    def parse_folder(self, folder_name):

        folder_path = self.root_dir / folder_name

        image_path = folder_path / "image.tif"

        image = cv2.imread(str(image_path))

        if image is None:
            print(f"[Skip] invalid image: {image_path}")
            return

        image_identifier = self.builder.register_image(
            str(image_path.relative_to(self.root_dir)),
            image.shape
        )

        mask_files = glob.glob(
            str(folder_path / "class*.tif")
        )

        for mask_file in mask_files:

            category = self.extract_category_id(mask_file)

            if category is None:
                continue

            self.builder.discovered_categories.add(category)

            self.process_mask(
                mask_file,
                image_identifier,
                category
            )

    def process_mask(
        self,
        mask_path,
        image_id,
        category_id
    ):

        mask = sio.imread(mask_path)

        if mask is None:
            return

        if mask.ndim == 3:

            mask = cv2.cvtColor(
                mask,
                cv2.COLOR_BGR2GRAY
            )

        instance_values = np.unique(mask)

        for instance_id in instance_values:

            if instance_id == 0:
                continue

            binary = (
                mask == instance_id
            ).astype(np.uint8)

            if binary.sum() == 0:
                continue

            polygons = PolygonExtractor.binary_to_segments(
                binary
            )

            for poly in polygons:

                self.builder.register_annotation(
                    image_id,
                    category_id,
                    poly
                )

    @staticmethod
    def extract_category_id(path):

        filename = os.path.basename(path)

        try:
            return int(
                filename
                .replace("class", "")
                .replace(".tif", "")
            )

        except:
            print(f"[Invalid category] {filename}")
            return None


# ============================================================
# Main
# ============================================================

def build_split_json(
    folders,
    root_dir,
    output_path
):

    parser = InstanceDatasetParser(root_dir)

    for folder_name in folders:
        parser.parse_folder(folder_name)

    parser.builder.export(output_path)


if __name__ == "__main__":

    dataset_root = "/mnt/HDD4/anlt/data/hw3-data copy"

    folder_pool = sorted([
        name for name in os.listdir(dataset_root)
        if os.path.isdir(
            os.path.join(dataset_root, name)
        )
    ])

    train_subset, val_subset = train_test_split(
        folder_pool,
        test_size=0.1,
        random_state=42
    )

    print(
        f"Train folders: {len(train_subset)} | "
        f"Val folders: {len(val_subset)}"
    )

    build_split_json(
        train_subset,
        dataset_root,
        "train.json"
    )

    build_split_json(
        val_subset,
        dataset_root,
        "val.json"
    )

    print(
        f"Generating full.json "
        f"using {len(folder_pool)} folders..."
    )

    build_split_json(
        folder_pool,
        dataset_root,
        "full.json"
    )