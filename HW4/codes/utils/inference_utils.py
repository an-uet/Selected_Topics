import torch
import torch.nn.functional as F


def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    H = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_, height, width


def tta_eval(model, input_):
    """8-fold TTA: 4 rotations x 2 flips, averaged predictions."""
    preds = []
    for rot in [0, 1, 2, 3]:
        for flip in [False, True]:
            x = torch.rot90(input_, rot, [2, 3])
            if flip:
                x = torch.flip(x, [3])
            pred = model(x)
            if flip:
                pred = torch.flip(pred, [3])
            pred = torch.rot90(pred, -rot, [2, 3])
            preds.append(pred)
    return torch.clamp(torch.stack(preds).mean(0), 0, 1)


def tile_eval(model, input_, tile=128, tile_overlap=32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)
            E[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch)
            W[..., h_idx:(h_idx + tile), w_idx:(w_idx + tile)].add_(out_patch_mask)

    return torch.clamp(E.div_(W), 0, 1)
