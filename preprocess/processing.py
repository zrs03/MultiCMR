"""Shared offline/online preprocessing; long-axis axis 0 is always time."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

VIEWS = ('2ch', '4ch', 'sa')
DEFAULT_OUTPUT_SIZES = {'2ch': (64, 128, 128), '4ch': (64, 128, 128),
                        'sa': (128, 128, 128)}


def validate_output_sizes(output_sizes=None):
    sizes = dict(DEFAULT_OUTPUT_SIZES if output_sizes is None else output_sizes)
    if set(sizes) != set(VIEWS):
        raise ValueError(f'output_sizes must contain {VIEWS}')
    for view, size in sizes.items():
        if len(size) != 3 or any(int(n) != n or n <= 0 for n in size):
            raise ValueError(f'{view}: expected three positive dimensions, got {size}')
        sizes[view] = tuple(int(n) for n in size)
    return sizes


def to_model_axes(array, view):
    """Raw [H,W,T] -> [T,H,W] for LAX; SAX [H,W,D] -> [D,H,W]."""
    if view not in VIEWS:
        raise ValueError(f'Unknown view: {view}')
    # Some long-axis NIfTIs explicitly encode the singleton slice dimension.
    if view != 'sa' and array.ndim == 4 and array.shape[2] == 1:
        array = array[:, :, 0, :]
    if array.ndim != 3:
        raise ValueError(f'{view}: expected a 3D array, got {array.shape}')
    return np.moveaxis(array, -1, 0).copy()


def resize_array(array, size, is_label=False):
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
    if is_label:
        out = F.interpolate(tensor, size=tuple(size), mode='nearest')
    else:
        out = F.interpolate(tensor, size=tuple(size), mode='trilinear', align_corners=False)
    return out[0, 0].numpy()


def prepare_nifti(path, view, output_size, is_label=False):
    """Resample time and space for LAX; labels always use nearest neighbors."""
    nii = nib.load(str(path))
    array = nii.get_fdata(dtype=np.float32)
    original_shape = array.shape
    array = to_model_axes(array, view)
    if is_label:
        array = np.rint(array)
    else:
        nonzero = array[np.abs(array) > 1e-8]
        if nonzero.size:
            array = (array - nonzero.mean()) / max(float(nonzero.std()), 1e-8)
    array = resize_array(array, output_size, is_label=is_label)
    return array.astype(np.int64 if is_label else np.float32), nii.affine, original_shape


def load_view(path, view, output_size, is_label=False):
    """Read a raw NIfTI or a preprocessed NPZ without transposing it twice."""
    if Path(path).suffix != '.npz':
        return prepare_nifti(path, view, output_size, is_label)
    with np.load(path, allow_pickle=False) as data:
        expected_layout = 'THW' if view != 'sa' else 'DHW'
        if str(data['view']) != view or str(data['layout']) != expected_layout:
            raise ValueError(f'Invalid view/layout in {path}')
        key = 'label' if is_label else 'image'
        if key not in data:
            raise ValueError(f'{path} has no {key}')
        array = data[key].copy()
        affine, original_shape = data['affine'].copy(), tuple(data['original_shape'])
    if tuple(array.shape) != tuple(output_size):
        raise ValueError(f'{path}: cached size {array.shape} != requested {tuple(output_size)}; '
                         'regenerate preprocessing with the requested per-view sizes')
    return array.astype(np.int64 if is_label else np.float32), affine, original_shape
