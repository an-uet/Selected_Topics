import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from utils.dataset_utils import TestSpecificDataset
from utils.image_io import save_image_tensor
from utils.inference_utils import pad_input, tta_eval, tile_eval
from train import PromptIRModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=3,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')
    parser.add_argument('--test_path', type=str,
                        default="/mnt/HDD4/anlt/data/hw4_realse_dataset/test/degraded/",
                        help='path to test images (directory or single image)')
    parser.add_argument('--output_path', type=str, default="output/", help='output save path')
    parser.add_argument('--ckpt_name', type=str, default="best_rainsnow_edge.ckpt",
                        help='checkpoint filename')
    parser.add_argument('--tta', action='store_true', help="Use 8-fold test-time augmentation")
    parser.add_argument('--tile', type=bool, default=False, help="Enable tiled inference")
    parser.add_argument('--tile_size', type=int, default=128, help='Tile size for tiled inference')
    parser.add_argument('--tile_overlap', type=int, default=32, help='Tile overlap for tiled inference')
    parser.add_argument('--ckpt_dir', type=str, default="train_ckpt/", help='Checkpoint directory')
    opt = parser.parse_args()

    ckpt_path = opt.ckpt_dir + opt.ckpt_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(opt.output_path, exist_ok=True)

    np.random.seed(0)
    torch.manual_seed(0)

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.cuda)
    print(f"Loading model {opt.ckpt_name}")
    net = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    net.eval()

    test_set = TestSpecificDataset(opt)
    testloader = DataLoader(test_set, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    print('Start testing...')
    with torch.no_grad():
        for ([clean_name], degrad_patch) in tqdm(testloader):
            degrad_patch = degrad_patch.to(device)

            if opt.tta:
                restored = tta_eval(net, degrad_patch)
            elif opt.tile is False:
                restored = net(degrad_patch)
            else:
                print("Using Tiling")
                degrad_patch, h, w = pad_input(degrad_patch)
                restored = tile_eval(net, degrad_patch,
                                     tile=opt.tile_size, tile_overlap=opt.tile_overlap)
                restored = restored[:, :, :h, :w]

            print(f"Saving {clean_name[0]}...")
            save_image_tensor(restored, opt.output_path + clean_name[0] + '.png')
