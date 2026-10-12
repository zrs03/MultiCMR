"""Small in-plane affine transforms shared across frames/slices and labels."""
import math

import numpy as np
import torch
import torch.nn.functional as F


def spatial_affine(image, label, degrees, translate_x, translate_y, scale):
    """K/D,H,W image; label H,W or D,H,W. Translation is fraction of W/H.

    A single inverse sampling grid applies to all frames/slices. No temporal
    resampling or SAX depth mixing. Positive translation moves content right/down.
    """
    height, width = image.shape[-2:]
    angle = math.radians(degrees)
    c, s = math.cos(angle) / scale, math.sin(angle) / scale
    # Rotation in pixel coordinates, corrected for non-square image dimensions.
    matrix = torch.tensor([[c, s * height / width], [-s * width / height, c]], dtype=torch.float32)
    offset = -(matrix @ torch.tensor([2 * translate_x, 2 * translate_y]))
    theta = torch.cat([matrix, offset[:, None]], dim=1)[None]
    tensor = torch.from_numpy(np.ascontiguousarray(image)).float()[None]
    grid = F.affine_grid(theta, tensor.shape, align_corners=False)
    transformed = F.grid_sample(tensor, grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0].numpy()
    target = None
    if label is not None:
        original_ndim = label.ndim
        labels = torch.from_numpy(np.ascontiguousarray(label)).float()
        labels = labels[None, None] if original_ndim == 2 else labels[None]
        target = F.grid_sample(labels, grid, mode='nearest', padding_mode='zeros', align_corners=False)[0].numpy()
        if original_ndim == 2:
            target = target[0]
        target = target.astype(label.dtype)
    return transformed, target


def random_spatial_affine(image, label, probability=.5, degrees=10., translate=.05, scale=.1):
    if np.random.rand() >= probability:
        return image, label
    return spatial_affine(image, label, np.random.uniform(-degrees, degrees),
                          np.random.uniform(-translate, translate), np.random.uniform(-translate, translate),
                          np.random.uniform(1 - scale, 1 + scale))
