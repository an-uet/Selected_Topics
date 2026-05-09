import os
import cv2
import json
import torch
import zipfile
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from pycocotools import mask as mask_utils

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer


os.environ["CUDA_VISIBLE_DEVICES"] = "4"


# ============================================================
# Argument Parser
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_folder",
        default="/mnt/HDD4/anlt/data/hw3-data/test_release"
    )

    parser.add_argument(
        "--output_dir",
        default="results/models/X101_FPN3x"
    )

    parser.add_argument(
        "--trained_model",
        default="./checkpoints/models/X101_FPN3x"
    )

    parser.add_argument(
        "--mapping_json",
        default="/mnt/HDD4/anlt/data/hw3-data/test_image_name_to_ids.json"
    )

    parser.add_argument(
        "--device",
        default="cuda"
    )

    return parser


# ============================================================
# Configuration
# ============================================================

class DetectronInferenceConfig:

    def __init__(self, model_dir):

        self.model_dir = model_dir

    def build(self):

        cfg = get_cfg()

        cfg.merge_from_file(
            os.path.join(
                self.model_dir,
                "config.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = os.path.join(
            self.model_dir,
            "model_best.pth"
        )

        cfg.TEST.DETECTIONS_PER_IMAGE = 2000

        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1500

        return cfg


# ============================================================
# Utility Functions
# ============================================================

def load_metadata(mapping_path):

    with open(mapping_path, "r") as fp:
        metadata = json.load(fp)

    file_to_id = {}
    file_to_size = {}

    for item in metadata:

        filename = item["file_name"]

        file_to_id[filename] = item["id"]

        file_to_size[filename] = (
            item["height"],
            item["width"]
        )

    return file_to_id, file_to_size


def encode_mask(binary_mask):

    encoded = mask_utils.encode(
        np.asfortranarray(
            binary_mask.astype(np.uint8)
        )
    )

    encoded["counts"] = encoded["counts"].decode("utf-8")

    return encoded


def export_visualization(
    image,
    predictions,
    save_path
):

    drawer = Visualizer(
        image[:, :, ::-1],
        scale=0.5
    )

    rendered = drawer.draw_instance_predictions(
        predictions
    )

    cv2.imwrite(
        str(save_path),
        rendered.get_image()[:, :, ::-1]
    )


# ============================================================
# Inference Pipeline
# ============================================================

class InstanceSegmentationInference:

    def __init__(self, args):

        self.args = args

        cfg_builder = DetectronInferenceConfig(
            args.trained_model
        )

        self.predictor = DefaultPredictor(
            cfg_builder.build()
        )

        self.image_to_id, self.image_sizes = load_metadata(
            args.mapping_json
        )

        self.results = []

        self._prepare_directories()

    def _prepare_directories(self):

        self.output_root = Path(
            self.args.output_dir
        )

        self.visualize_root = (
            self.output_root / "visualize"
        )

        self.output_root.mkdir(
            parents=True,
            exist_ok=True
        )

        self.visualize_root.mkdir(
            parents=True,
            exist_ok=True
        )

    def collect_test_images(self):

        return sorted([
            file_name
            for file_name in os.listdir(
                self.args.test_folder
            )
            if file_name.lower().endswith(
                (".png", ".tif")
            )
        ])

    def run(self):

        test_images = self.collect_test_images()

        for filename in tqdm(
            test_images,
            desc="Running inference"
        ):

            if filename not in self.image_to_id:
                continue

            self.process_single_image(filename)

        self.export_outputs()

    def process_single_image(self, filename):

        image_path = os.path.join(
            self.args.test_folder,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            return

        prediction = self.predictor(image)

        instances = prediction["instances"].to("cpu")

        self.save_visual_result(
            image,
            instances,
            filename
        )

        self.convert_instances_to_coco(
            filename,
            instances
        )

    def save_visual_result(
        self,
        image,
        instances,
        filename
    ):

        output_path = (
            self.visualize_root /
            filename.replace(".tif", ".png")
        )

        export_visualization(
            image,
            instances,
            output_path
        )

    def convert_instances_to_coco(
        self,
        filename,
        instances
    ):

        image_id = self.image_to_id[filename]

        H, W = self.image_sizes[filename]

        boxes = instances.pred_boxes.tensor.numpy()

        masks = instances.pred_masks.numpy()

        scores = instances.scores.numpy()

        labels = instances.pred_classes.numpy()

        for box, mask, score, cls_id in zip(
            boxes,
            masks,
            scores,
            labels
        ):

            x1, y1, x2, y2 = box

            rle = encode_mask(mask)

            annotation = {

                "image_id":
                    image_id,

                "bbox":
                    [
                        float(x1),
                        float(y1),
                        float(x2 - x1),
                        float(y2 - y1)
                    ],

                "score":
                    float(score),

                "category_id":
                    int(cls_id) + 1,

                "segmentation": {
                    "size": [H, W],
                    "counts": rle["counts"]
                }
            }

            self.results.append(annotation)

    def export_outputs(self):

        self.results = sorted(
            self.results,
            key=lambda x: x["image_id"]
        )

        json_path = (
            self.output_root /
            "test-results.json"
        )

        with open(json_path, "w") as fp:
            json.dump(self.results, fp)

        zip_path = (
            self.output_root /
            f"{self.output_root.name}.zip"
        )

        with zipfile.ZipFile(
            zip_path,
            "w",
            zipfile.ZIP_DEFLATED
        ) as archive:

            archive.write(
                json_path,
                arcname=json_path.name
            )

        print(f"Saved JSON : {json_path}")
        print(f"Saved ZIP  : {zip_path}")


# ============================================================
# Main
# ============================================================

def main():

    args = build_parser().parse_args()

    pipeline = InstanceSegmentationInference(args)

    pipeline.run()


if __name__ == "__main__":
    main()