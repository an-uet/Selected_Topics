#!/usr/bin/env python3
"""Simple single-image inference script for Deformable-DETR.

Usage example:
python inference_single.py --image /path/to/img.jpg --checkpoint /path/to/checkpoint.pth --device cuda --output out.jpg

This script imports the repo model builder, loads a checkpoint, runs the model on one image,
applies the bbox postprocessor, prints detections and optionally saves a visualization.
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T


def parse_args():
    parser = argparse.ArgumentParser("Deformable DETR single-image inference")
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--device', default='cuda', help='cpu or cuda')
    parser.add_argument('--out', default='', help='Path to save visualized output (optional)')
    parser.add_argument('--score-thr', type=float, default=0.3, help='Score threshold to display boxes')
    return parser.parse_args()


def main():
    args = parse_args()

    # Import the repo model factory and default args parser
    try:
        from main import get_args_parser as _get_main_parser
    except Exception:
        print('Failed to import `main.get_args_parser`. Are you running from the repo root?')
        raise

    # Build a default args namespace for model construction (mirrors training defaults)
    main_parser = _get_main_parser()
    main_args = main_parser.parse_args([])
    # Force eval/device settings
    main_args.device = args.device
    main_args.batch_size = 1

    # Build model
    try:
        from models import build_model
    except Exception as e:
        print('Failed to import model builder from `models`.')
        raise

    model, criterion, postprocessors = build_model(main_args)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    if 'model' in ckpt:
        state_dict = ckpt['model']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=False)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model.to(device)
    model.eval()

    # Load image and apply minimal preprocessing (matches common DETR preprocessing)
    img = Image.open(args.image).convert('RGB')
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    tensor = transform(img)

    # Run model
    with torch.no_grad():
        outputs = model([tensor.to(device)])

        # postprocessor expects target sizes (H, W)
        target_sizes = torch.tensor([[img.height, img.width]], device=device)
        results = postprocessors['bbox'](outputs, target_sizes)

    if not isinstance(results, (list, tuple)):
        print('Unexpected postprocessor output; printing raw outputs')
        print(results)
        return

    res = results[0]
    boxes = res['boxes'].cpu()
    scores = res['scores'].cpu()
    labels = res['labels'].cpu()

    # Print top detections
    print('Detections:')
    for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
        if s.item() < args.score_thr:
            continue
        x1, y1, x2, y2 = b.tolist()
        print(f'  {i}: label={int(l.item())} score={s.item():.3f} box={[x1,y1,x2,y2]}')

    # Optional visualization
    if args.out:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for b, s, l in zip(boxes, scores, labels):
            if s.item() < args.score_thr:
                continue
            x1, y1, x2, y2 = b.tolist()
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
            caption = f'{int(l.item())}: {s.item():.2f}'
            draw.text((x1 + 3, y1 + 3), caption, fill='white', font=font)
        out_path = Path(args.out)
        img.save(out_path)
        print(f'Saved visualization to {out_path}')


if __name__ == '__main__':
    main()
