"""Native-frame spatial preprocessing and centered-window cache loading."""
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

VIEWS = ('2ch', '4ch', 'sa')
DEFAULT_OUTPUT_SIZES = {'2ch': (5, 128, 128), '4ch': (5, 128, 128),
                        'sa': (5, 128, 128)}


def validate_output_sizes(output_sizes=None):
    sizes = dict(DEFAULT_OUTPUT_SIZES if output_sizes is None else output_sizes)
    if set(sizes) != set(VIEWS):
        raise ValueError(f'output_sizes must contain {VIEWS}')
    for view, size in sizes.items():
        if len(size) != 3 or any(int(n) != n or n <= 0 for n in size):
            raise ValueError(f'{view}: expected three positive dimensions, got {size}')
        sizes[view] = tuple(int(n) for n in size)
    if sizes['2ch'] != sizes['4ch'] or sizes['2ch'][0] % 2 != 1:
        raise ValueError('LAX sizes must agree and use a positive odd window length')
    if sizes['sa'][0] != 5:
        raise ValueError('SAX context must have 5 channels: center, t-1, t+1, z-1, z+1')
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
    """Preserve all native frames; resize H/W only, including flattened SAX."""
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
    # Never interpolate the flattened slice/phase axis.
    tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[:, None]
    kwargs = {} if is_label else {'align_corners': False}
    array = F.interpolate(tensor, size=tuple(output_size[-2:]),
                          mode='nearest' if is_label else 'bilinear', **kwargs)[:, 0].numpy()
    return array.astype(np.int64 if is_label else np.float32), nii.affine, original_shape


def load_view(path, view, output_size, is_label=False, complete_cine=False):
    """Read a context/full NPZ with an explicit layout."""
    if Path(path).suffix != '.npz':
        raise ValueError('SAX context schema requires offline NPZ caches with phase metadata; rerun preprocessing')
    with np.load(path, allow_pickle=False) as data:
        expected_layout = ('FHW' if view == 'sa' else 'THW') if complete_cine else 'KHW'
        if str(data['view']) != view or str(data['layout']) != expected_layout:
            raise ValueError(f'Invalid view/layout in {path}')
        key = 'label' if is_label else 'image'
        if key not in data:
            raise ValueError(f'{path} has no {key}')
        array = data[key].copy()
        affine, original_shape = data['affine'].copy(), tuple(data['original_shape'])
    expected = (array.shape[0], *output_size[-2:]) if complete_cine else (output_size[-2:] if is_label else output_size)
    if tuple(array.shape) != tuple(expected):
        raise ValueError(f'{path}: cached size {array.shape} != requested {tuple(output_size)}; '
                         'regenerate preprocessing with the requested per-view sizes')
    return array.astype(np.int64 if is_label else np.float32), affine, original_shape


def window_indices(center, length, window_size):
    if window_size < 1 or window_size % 2 != 1 or not 0 <= center < length:
        raise ValueError('Invalid center, sequence length or odd window size')
    radius = window_size // 2
    return np.clip(np.arange(center - radius, center + radius + 1), 0, length - 1)


def paired_centers(lengths):
    """Nearest normalized sequence positions; not a physiological phase alignment."""
    count = max(lengths.values())
    return [{v: int(np.rint(i * (n - 1) / max(count - 1, 1))) for v, n in lengths.items()}
            for i in range(count)]


def validate_sax_geometry(frames, phases):
    if isinstance(phases, (bool, np.bool_)) or not isinstance(phases, (int, np.integer)) or phases < 1:
        raise ValueError('SAX phase count must be a positive integer from metadata')
    if frames < 1 or frames % phases:
        raise ValueError(f'SAX frame count {frames} is not divisible by phase count {phases}')
    return frames // phases


def reflect_index(index, length):
    if length < 1:
        raise ValueError('Cannot reflect an empty axis')
    if length == 1:
        return 0
    index %= 2 * (length - 1)
    return min(index, 2 * (length - 1) - index)


def sax_context_indices(t, z, phases, slices):
    """Canonical flat layout k=z*T+t. Order is center, t-, t+, z-, z+."""
    if not 0 <= t < phases or not 0 <= z < slices:
        raise ValueError('SAX center outside slice/phase range')
    return np.array([z * phases + t,
                     z * phases + reflect_index(t - 1, phases),
                     z * phases + reflect_index(t + 1, phases),
                     reflect_index(z - 1, slices) * phases + t,
                     reflect_index(z + 1, slices) * phases + t], dtype=np.int64)
