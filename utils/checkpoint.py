"""Epoch-boundary checkpoints, including per-rank random state."""
import os
import random
import warnings

import numpy as np
import torch

ARCHITECTURE = 'center_window_film_lax_3d_sax_v2'


def capture_rng_state():
    state = {'python': random.getstate(), 'numpy': np.random.get_state(),
             'torch': torch.get_rng_state()}
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def restore_training_state(checkpoint, model, optimizer, scheduler, rank=0, world_size=1):
    """Return next epoch and historical best; older weights support warm starts."""
    state = {k.removeprefix('module.'): v for k, v in checkpoint['model'].items()}
    model.load_state_dict(state, strict=True)
    required = ('optimizer', 'scheduler', 'epoch', 'best_val_dice')
    if not all(k in checkpoint for k in required):
        if rank == 0:
            warnings.warn('Checkpoint contains weights only: optimizer/scheduler and epoch cannot be '
                          'restored. Starting from epoch 1 with loaded weights.', stacklevel=2)
        return 1, -1.0
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    states = checkpoint.get('rng_states', [])
    if len(states) == world_size:
        state = states[rank]
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if torch.cuda.is_available() and 'cuda' in state:
            torch.cuda.set_rng_state(state['cuda'].cpu())
    elif rank == 0:
        warnings.warn('Random states unavailable or world size changed; using the current seeded '
                      'RNG states. Training resumes, but will not exactly replay the original run.', stacklevel=2)
    return int(checkpoint['epoch']) + 1, float(checkpoint['best_val_dice'])


def atomic_save(checkpoint, path):
    """Keep the previous checkpoint intact until the new file is fully written."""
    temporary = str(path) + '.tmp'
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
