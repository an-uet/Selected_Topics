"""
Model Ensemble + TTA Inference for MSW-DAT
==========================================
Loads multiple checkpoints, runs inference with 8-augmentation TTA on each,
then averages all outputs for the final prediction.

Usage:
    python scripts/ensemble_infer.py [--gpu GPU_ID] [--no-tta] [--out-dir DIR]

Output: SR images saved to --out-dir, then run gen.py separately.
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Make sure basicsr is importable from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basicsr.archs.dat_arch import DAT
from basicsr.archs.wavedat_arch import WaveDAT


# ─── Geometric augmentations (8 transforms) ──────────────────────────────────

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


# ─── Model loading ────────────────────────────────────────────────────────────

ARCH_KWARGS = dict(
    upscale=4, in_chans=3, img_size=64, img_range=1.,
    split_size=[8, 8], depth=[6, 6, 6, 6, 6, 6],
    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
    expansion_factor=4, resi_connection='1conv',
)

def load_model(ckpt_path, device, arch='DAT'):
    """arch: 'DAT' or 'WaveDAT'"""
    cls = WaveDAT if arch == 'WaveDAT' else DAT
    model = cls(**ARCH_KWARGS).to(device)
    state = torch.load(ckpt_path, map_location=device)
    weights = state.get('params_ema', state.get('params', state))
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


# ─── Inference helpers ────────────────────────────────────────────────────────

@torch.no_grad()
def infer_single(model, lq, use_tta):
    """lq: (1, 3, H, W) float32 tensor on device. Returns (1, 3, 4H, 4W)."""
    if use_tta:
        preds = []
        for mode in range(8):
            aug = _augment(lq, mode)
            out = model(aug)
            preds.append(_inverse_augment(out, mode))
        return torch.stack(preds).mean(0)
    else:
        return model(lq)


def read_lq(path, device):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return t

def save_sr(tensor, path):
    img = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    img = (img * 255).round().astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--lq-dir',  default='/mnt/HDD4/anlt/data/test/lr',
                   help='Input LR image folder')
    p.add_argument('--out-dir', default='results/MSW_ensemble/visualization/Single',
                   help='Output SR image folder')
    p.add_argument('--gpu',     default='0', help='CUDA_VISIBLE_DEVICES')
    p.add_argument('--no-tta',  action='store_true', help='Disable TTA')
    p.add_argument('--weights', nargs='+',
                   help='Checkpoint paths. If omitted, uses default ensemble.')
    return p.parse_args()


DEFAULT_CHECKPOINTS = [
    # (path, weight, arch)
    # MSW_charb_fft: 31.292 dB val → 34.330 LB
    ('experiments/MSW_finetune_charb_fft/models/net_g_115000.pth', 1, 'DAT'),
    # WaveDAT: 31.3576 dB val @ 285K — best
    ('experiments/WaveDAT/models/net_g_285000.pth',                1, 'WaveDAT'),
]


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_tta = not args.no_tta

    # Resolve project root (scripts/ is one level below root)
    root = Path(__file__).resolve().parents[1]

    # Build checkpoint list
    if args.weights:
        ckpt_list = [(root / w, 1, 'DAT') for w in args.weights]
    else:
        ckpt_list = [(root / p, w, arch) for p, w, arch in DEFAULT_CHECKPOINTS
                     if (root / p).exists()]

    if not ckpt_list:
        raise RuntimeError('No checkpoints found. Check paths in DEFAULT_CHECKPOINTS.')

    print(f'Ensemble: {len(ckpt_list)} model(s), TTA={use_tta}')
    for p, w, arch in ckpt_list:
        print(f'  weight={w}  arch={arch}  {p.name}')

    # Load models
    models = []
    for ckpt_path, weight, arch in ckpt_list:
        print(f'Loading {ckpt_path.name} ({arch}) ...', end=' ', flush=True)
        m = load_model(ckpt_path, device, arch=arch)
        models.append((m, weight))
        print('done')

    total_weight = sum(w for _, w in models)

    # Output dir
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test images
    lq_dir = Path(args.lq_dir)
    img_paths = sorted(lq_dir.glob('*.png')) + sorted(lq_dir.glob('*.jpg'))
    print(f'\nProcessing {len(img_paths)} images → {out_dir}')

    for img_path in tqdm(img_paths):
        lq = read_lq(img_path, device)

        # Weighted average of all model outputs
        acc = None
        for model, weight in models:
            pred = infer_single(model, lq, use_tta)
            if acc is None:
                acc = pred * weight
            else:
                acc = acc + pred * weight

        sr = acc / total_weight

        # Save with '_x4' suffix to match gen.py expectation
        stem = img_path.stem
        out_path = out_dir / f'{stem}_x4.png'
        save_sr(sr, out_path)

    print(f'\nDone. SR images saved to {out_dir}')
    print('\nNext step — generate submission CSV:')
    print(f'  python gen.py -f {out_dir} -s results/MSW_ensemble.csv')


if __name__ == '__main__':
    main()
