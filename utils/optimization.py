"""AdamW with explicit decay groups for the segmentation model."""
import math

import torch


def build_adamw(model, lr=1e-4, weight_decay=1e-4):
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('lr must be positive and weight_decay nonnegative and finite')
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # Includes attention in_proj_weight/out_proj.weight. Excludes biases,
        # normalization, scalars, positional embeddings and anatomy tokens.
        is_weight = name == 'weight' or name.endswith('.weight') or name.endswith('.in_proj_weight')
        (decay if is_weight and parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.},
    ], lr=lr)
