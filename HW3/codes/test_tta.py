import os
import cv2
import json
import torch
import zipfile
import argparse
import numpy as np

from tqdm import tqdm
from pycocotools import mask as mask_utils

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.layers import nms
from detectron2.structures import Boxes, Instances
from detectron2.utils.visualizer import Visualizer

from mpvit import add_mpvit_config


os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# ============================================================
# Argument
# ============================================================

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_folder",
        default="/mnt/HDD4/anlt/data/hw3-data/test_release"
    )

    parser.add_argument(
        "--output_dir",
        default="results/"
    )

    parser.add_argument(
        "--trained_model",
        default="checkpoint/"
    )

    parser.add_argument(
        "--mapping_json",
        default="/mnt/HDD4/anlt/data/hw3-data/test_image_name_to_ids.json"
    )

    return parser


# ============================================================
# Augmentation Engine
# ============================================================

class GeometryTransform:

    def __init__(self):

        self.forward_ops = {
            "none": lambda x: x,
            "flip_h": lambda x: cv2.flip(x, 1),
            "flip_v": lambda x: cv2.flip(x, 0),
            "r90": lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
            "r180": lambda x: cv2.rotate(x, cv2.ROTATE_180),
            "r270": lambda x: cv2.rotate(x, cv2.ROTATE_90_COUNTERCLOCKWISE),
        }

    def apply(self, image, mode):
        return self.forward_ops[mode](image)

    def restore_mask(self, mask, mode):

        inverse_map = {
            "none": lambda x: x,
            "flip_h": lambda x: np.flip(x, axis=1),
            "flip_v": lambda x: np.flip(x, axis=0),
            "r90": lambda x: np.rot90(x, 1),
            "r180": lambda x: np.rot90(x, 2),
            "r270": lambda x: np.rot90(x, 3),
        }

        return inverse_map[mode](mask)

    def restore_points(self, pts, mode, h, w):

        x = pts[:, 0]
        y = pts[:, 1]

        transform_map = {
            "none": lambda: np.stack([x, y], axis=1),

            "flip_h": lambda: np.stack(
                [w - x, y], axis=1
            ),

            "flip_v": lambda: np.stack(
                [x, h - y], axis=1
            ),

            "r90": lambda: np.stack(
                [y, h - x], axis=1
            ),

            "r180": lambda: np.stack(
                [w - x, h - y], axis=1
            ),

            "r270": lambda: np.stack(
                [w - y, x], axis=1
            )
        }

        return transform_map[mode]()

    def restore_boxes(self, boxes, mode, h, w):

        if len(boxes) == 0:
            return boxes

        corners = np.stack([
            boxes[:, [0, 1]],
            boxes[:, [2, 1]],
            boxes[:, [2, 3]],
            boxes[:, [0, 3]]
        ], axis=1)

        flat = corners.reshape(-1, 2)

        restored = self.restore_points(flat, mode, h, w)

        restored = restored.reshape(-1, 4, 2)

        x1 = restored[:, :, 0].min(1)
        y1 = restored[:, :, 1].min(1)

        x2 = restored[:, :, 0].max(1)
        y2 = restored[:, :, 1].max(1)

        return np.stack([x1, y1, x2, y2], axis=1)


# ============================================================
# Predictor Wrapper
# ============================================================

class MPViTInference:

    def __init__(self, args):

        self.args = args

        self.predictor = self._initialize_model()

        self.transformer = GeometryTransform()

        self.tta_modes = [
            "none",
            "flip_h",
            "flip_v",
            "r90",
            "r180",
            "r270"
        ]

    def _initialize_model(self):

        cfg = get_cfg()

        add_mpvit_config(cfg)

        cfg.merge_from_file(
            os.path.join(
                self.args.trained_model,
                "config.yaml"
            )
        )

        cfg.MODEL.WEIGHTS = os.path.join(
            self.args.trained_model,
            "model_best.pth"
        )

        cfg.TEST.DETECTIONS_PER_IMAGE = 3000

        return DefaultPredictor(cfg)

    def predict_single(self, image, H, W):

        collected = {
            "boxes": [],
            "masks": [],
            "scores": [],
            "classes": []
        }

        for mode in self.tta_modes:

            transformed = self.transformer.apply(image, mode)

            outputs = self.predictor(transformed)

            instances = outputs["instances"].to("cpu")

            if len(instances) == 0:
                continue

            boxes = instances.pred_boxes.tensor.numpy()
            masks = instances.pred_masks.numpy()

            boxes = self.transformer.restore_boxes(
                boxes,
                mode,
                H,
                W
            )

            masks = np.stack([
                self.transformer.restore_mask(m, mode)
                for m in masks
            ])

            collected["boxes"].append(boxes)
            collected["masks"].append(masks)

            collected["scores"].append(
                instances.scores.numpy()
            )

            collected["classes"].append(
                instances.pred_classes.numpy()
            )

        return self.merge_predictions(
            collected,
            H,
            W
        )

    def merge_predictions(self, data, H, W,
                          iou_thr=0.5,
                          max_det=1000):

        if len(data["boxes"]) == 0:
            return Instances((H, W))

        boxes = torch.tensor(
            np.concatenate(data["boxes"]),
            dtype=torch.float32
        )

        masks = torch.tensor(
            np.concatenate(data["masks"]),
            dtype=torch.uint8
        )

        scores = torch.tensor(
            np.concatenate(data["scores"]),
            dtype=torch.float32
        )

        labels = torch.tensor(
            np.concatenate(data["classes"]),
            dtype=torch.int64
        )

        final_keep = []

        for cls_id in labels.unique():

            idx = (labels == cls_id).nonzero().squeeze(1)

            keep = nms(
                boxes[idx],
                scores[idx],
                iou_thr
            )

            final_keep.append(idx[keep])

        if len(final_keep) == 0:
            return Instances((H, W))

        final_keep = torch.cat(final_keep)

        ranking = torch.argsort(
            scores[final_keep],
            descending=True
        )

        final_keep = final_keep[ranking][:max_det]

        merged = Instances((H, W))

        merged.pred_boxes = Boxes(boxes[final_keep])
        merged.pred_masks = masks[final_keep].bool()

        merged.scores = scores[final_keep]
        merged.pred_classes = labels[final_keep]

        return merged


# ============================================================
# Utilities
# ============================================================

def read_mapping(path):

    with open(path) as f:
        content = json.load(f)

    image_to_id = {}
    image_to_size = {}

    for item in content:

        image_to_id[item["file_name"]] = item["id"]

        image_to_size[item["file_name"]] = (
            item["height"],
            item["width"]
        )

    return image_to_id, image_to_size


def encode_binary_mask(mask):

    rle = mask_utils.encode(
        np.asfortranarray(
            mask.astype(np.uint8)
        )
    )

    rle["counts"] = rle["counts"].decode("utf-8")

    return rle


def export_visualization(image, instances, save_path):

    vis = Visualizer(
        image[:, :, ::-1],
        scale=0.5
    )

    rendered = vis.draw_instance_predictions(instances)

    cv2.imwrite(
        save_path,
        rendered.get_image()[:, :, ::-1]
    )


# ============================================================
# Main
# ============================================================

def main():

    args = build_parser().parse_args()

    engine = MPViTInference(args)

    os.makedirs(args.output_dir, exist_ok=True)

    vis_dir = os.path.join(
        args.output_dir,
        "visualize"
    )

    os.makedirs(vis_dir, exist_ok=True)

    image_to_id, image_to_size = read_mapping(
        args.mapping_json
    )

    all_predictions = []

    image_files = sorted([
        f for f in os.listdir(args.test_folder)
        if f.endswith((".png", ".tif"))
    ])

    for fname in tqdm(image_files):

        if fname not in image_to_id:
            continue

        image_path = os.path.join(
            args.test_folder,
            fname
        )

        image = cv2.imread(image_path)

        H, W = image_to_size[fname]

        predictions = engine.predict_single(
            image,
            H,
            W
        )

        export_visualization(
            image,
            predictions,
            os.path.join(
                vis_dir,
                fname.replace(".tif", ".png")
            )
        )

        for box, mask, score, cls_id in zip(
            predictions.pred_boxes.tensor.numpy(),
            predictions.pred_masks.numpy(),
            predictions.scores.numpy(),
            predictions.pred_classes.numpy()
        ):

            x1, y1, x2, y2 = box

            rle = encode_binary_mask(mask)

            all_predictions.append({

                "image_id":
                    image_to_id[fname],

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
            })

    all_predictions = sorted(
        all_predictions,
        key=lambda x: x["image_id"]
    )

    json_path = os.path.join(
        args.output_dir,
        "test-results.json"
    )

    with open(json_path, "w") as f:
        json.dump(all_predictions, f)

    zip_path = os.path.join(
        args.output_dir,
        os.path.basename(args.output_dir) + ".zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zf:

        zf.write(
            json_path,
            arcname=os.path.basename(json_path)
        )

    print(f"Saved JSON -> {json_path}")
    print(f"Saved ZIP  -> {zip_path}")


if __name__ == "__main__":
    main()