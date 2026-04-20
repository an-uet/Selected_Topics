# inference.py
import argparse
import json
from pathlib import Path
import torch
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model
import torch.nn as nn
import util.misc as utils

def to_coco_results(outputs, image_ids):
    results = []
    for out, img_id in zip(outputs, image_ids):
        scores = out['scores'].tolist()
        labels = out['labels'].tolist()
        boxes = out['boxes'].tolist()  # xyxy absolute pixel coords
        for s, l, b in zip(scores, labels, boxes):
            x1, y1, x2, y2 = b
            width = x2 - x1
            height = y2 - y1
            results.append({
                'image_id': int(img_id),
                'category_id': int(l),
                'score': float(s),
                'bbox': [float(x1), float(y1), float(width), float(height)]
            })
    return results

def main(checkpoint_path, coco_path, output_json, device='cpu'):
    args = argparse.Namespace(
                lr=1e-4,
                lr_backbone=1e-5,
                batch_size=32,
                weight_decay=1e-4,
                epochs=1,
                lr_drop=200,
                clip_max_norm=0.1,
                frozen_weights=None,
                backbone='resnet50',
                dilation=False,
                position_embedding='sine',
                enc_layers=6,
                dec_layers=6,
                dim_feedforward=2048,
                hidden_dim=256,
                dropout=0.1,
                nheads=8,
                num_queries=100,
                pre_norm=False,
                masks=False,
                no_aux_loss=False,
                aux_loss=True,
                set_cost_class=1,
                set_cost_bbox=5,
                set_cost_giou=2,
                mask_loss_coef=1,
                dice_loss_coef=1,
                bbox_loss_coef=5,
                giou_loss_coef=2,
                eos_coef=0.1,
                dataset_file='coco',
                coco_path=coco_path,
                coco_panoptic_path=None,
                remove_difficult=False,
                output_dir='',
                device=device,
                seed=42,
                resume='',
                start_epoch=0,
                eval=False,
                num_workers=2,
                world_size=1,
                dist_url='env://',
                distributed=False,
            )

    torch_device = torch.device(device)
    model, criterion, postprocessors = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)

    class_key = None
    for k in state_dict.keys():
        if k.endswith('class_embed.weight'):
            class_key = k
            break

    if class_key is not None:
        ckpt_num_classes_plus1 = state_dict[class_key].shape[0]
        cur_num = model.class_embed.out_features
        if ckpt_num_classes_plus1 != cur_num:
            # replace model.class_embed to match checkpoint
            in_features = model.class_embed.in_features
            model.class_embed = nn.Linear(in_features, ckpt_num_classes_plus1)

    model.load_state_dict(state_dict)
    model.to(torch_device)
    model.eval()

    try:
        dataset = build_dataset('test', args)
    except Exception:
        print("Warning: 'test' split not found or cannot be loaded. Falling back to images-only loader.")
        from datasets.coco import make_coco_transforms
        from PIL import Image

        class ImagesOnlyDataset(torch.utils.data.Dataset):
            def __init__(self, img_folder, transforms=None):
                self.img_folder = Path(img_folder)
                print(f"Loading images from {self.img_folder}...")
                exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                self.files = [p for p in sorted(self.img_folder.rglob('*')) if p.suffix.lower() in exts]
                self.transforms = transforms

            def __len__(self):
                return len(self.files)

            def __getitem__(self, idx):
                p = self.files[idx]
                img = Image.open(p).convert('RGB')
                w, h = img.size
                # create a minimal target similar to CocoDetection output
                target = {'image_id': torch.tensor([int(p.stem) if p.stem.isdigit() else idx]),
                          'annotations': [],
                          'orig_size': torch.as_tensor([h, w]),
                          'size': torch.as_tensor([h, w])}
                if self.transforms is not None:
                    img, target = self.transforms(img, target)
                return img, target

        transforms = make_coco_transforms('val')
        test_folder = Path(args.coco_path) / 'test'
        if not test_folder.exists():
            # try a plain 'test' at provided path
            test_folder = Path(args.coco_path)
        dataset = ImagesOnlyDataset(test_folder, transforms=transforms)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False,
                                              collate_fn=utils.collate_fn, num_workers=2)

    all_results = []
    image_ids = []
    total = len(dataset)
    with torch.no_grad():
        for i, (samples, targets) in enumerate(data_loader, 1):
            display = None
            try:
                if hasattr(dataset, 'files'):
                    display = str(dataset.files[i-1])
                elif hasattr(dataset, 'coco'):
                    img_id = int(targets[0]['image_id'].item())
                    try:
                        fname = dataset.coco.loadImgs(img_id)[0]['file_name']
                    except Exception:
                        fname = dataset.coco.imgs[img_id]['file_name']
                    root = getattr(dataset, 'root', None)
                    display = str(Path(root) / fname) if root is not None else fname
                else:
                    display = f"image_id={int(targets[0]['image_id'].item())}"
            except Exception:
                display = f"image_id={int(targets[0]['image_id'].item())}"
            print(f'Processing [{i}/{total}]: {display}')

            samples = samples.to(torch_device)
            orig_sizes = torch.stack([t['orig_size'] for t in targets], dim=0).to(torch_device)
            outputs = model(samples)
            results = postprocessors['bbox'](outputs, orig_sizes)
            # results is a list (len=batch) of dicts with 'scores','labels','boxes'
            img_ids = [int(t['image_id'].item()) for t in targets]
            all_results.extend(results)
            image_ids.extend(img_ids)

    coco_results = to_coco_results(all_results, image_ids)
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(coco_results, f)
    print('Saved', output_json)

if __name__ == '__main__':
    import sys
    ckpt = sys.argv[1]
    coco_path = sys.argv[2]
    out = sys.argv[3]
    device = sys.argv[4] if len(sys.argv) > 4 else 'cpu'
    main(ckpt, coco_path, out, device)
