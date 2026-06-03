"""
Stochastic Weight Averaging (SWA) Inference
============================================
Averages model weights across multiple checkpoints into a single "SWA model",
then runs TTA inference. Often outperforms output-ensemble because weight
averaging smooths the loss surface, not just the predictions.

Usage:
    python scripts/swa_infer.py [--gpu GPU] [--no-tta] [--out-dir DIR]
"""

import argparse
import os
import sys
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.archs.dat_arch import DAT


ARCH_KWARGS = dict(
    upscale=4, in_chans=3, img_size=64, img_range=1.,
    split_size=[8, 8], depth=[6, 6, 6, 6, 6, 6],
    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    expansion_factor=4, resi_connection='1conv',
)

# Checkpoints to average — latest checkpoints have best individual quality
DEFAULT_SWA_CKPTS = [
    'experiments/MSW/models/net_g_475000.pth',
    'experiments/MSW/models/net_g_470000.pth',
    'experiments/MSW/models/net_g_465000.pth',
    'experiments/MSW/models/net_g_460000.pth',
    'experiments/MSW/models/net_g_455000.pth',
    'experiments/MSW/models/net_g_450000.pth',
]


def _augment(img, mode):
    if mode == 0: return img
    if mode == 1: return img.flip(-1)
    if mode == 2: return img.flip(-2)
    if mode == 3: return img.transpose(-2, -1).flip(-1)
    if mode == 4: return img.flip(-1).flip(-2)
    if mode == 5: return img.transpose(-2, -1).flip(-2)
    if mode == 6: return img.flip(-1).transpose(-2, -1)
    if mode == 7: return img.transpose(-2, -1)

def _inverse_augment(img, mode):
    if mode == 0: return img
    if mode == 1: return img.flip(-1)
    if mode == 2: return img.flip(-2)
    if mode == 3: return img.flip(-1).transpose(-2, -1)
    if mode == 4: return img.flip(-2).flip(-1)
    if mode == 5: return img.flip(-2).transpose(-2, -1)
    if mode == 6: return img.transpose(-2, -1).flip(-1)
    if mode == 7: return img.transpose(-2, -1)


def build_swa_model(ckpt_paths, device):
    """Average weights of multiple checkpoints into one model."""
    print(f'Building SWA model from {len(ckpt_paths)} checkpoints...')
    avg_state = None
    for i, path in enumerate(ckpt_paths):
        print(f'  [{i+1}/{len(ckpt_paths)}] Loading {Path(path).name}')
        state = torch.load(path, map_location='cpu')
        weights = state.get('params_ema', state.get('params', state))
        if avg_state is None:
            avg_state = OrderedDict({k: v.float() for k, v in weights.items()})
        else:
            for k in avg_state:
                avg_state[k] += weights[k].float()

    # Average
    n = len(ckpt_paths)
    for k in avg_state:
        avg_state[k] /= n

    model = DAT(**ARCH_KWARGS).to(device)
    model.load_state_dict(avg_state, strict=True)
    model.eval()
    print('SWA model ready.')
    return model


def read_lq(path, device):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

def save_sr(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


@torch.no_grad()
def infer(model, lq, use_tta):
    if use_tta:
        preds = [_inverse_augment(model(_augment(lq, m)), m) for m in range(8)]
        return torch.stack(preds).mean(0)
    return model(lq)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lq-dir',  default='/mnt/HDD4/anlt/data/test/lr')
    p.add_argument('--out-dir', default='results/MSW_swa/visualization/Single')
    p.add_argument('--gpu',     default='0')
    p.add_argument('--no-tta',  action='store_true')
    p.add_argument('--ckpts',   nargs='+', help='Override checkpoint list')
    return p.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root = Path(__file__).resolve().parents[1]

    ckpt_paths = [str(root / c) for c in (args.ckpts or DEFAULT_SWA_CKPTS)]
    ckpt_paths = [p for p in ckpt_paths if Path(p).exists()]
    if not ckpt_paths:
        raise RuntimeError('No checkpoints found.')

    model = build_swa_model(ckpt_paths, device)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lq_dir = Path(args.lq_dir)
    img_paths = sorted(lq_dir.glob('*.png')) + sorted(lq_dir.glob('*.jpg'))
    use_tta = not args.no_tta
    print(f'Processing {len(img_paths)} images | TTA={use_tta} | → {out_dir}')

    for img_path in tqdm(img_paths):
        sr = infer(model, read_lq(img_path, device), use_tta)
        save_sr(sr, out_dir / f'{img_path.stem}_x4.png')

    print(f'\nDone. Next:')
    print(f'  python gen.py -f {out_dir} -s results/MSW_swa.csv')


if __name__ == '__main__':
    main()
