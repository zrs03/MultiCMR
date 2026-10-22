"""2D MedNeXt-M stages using the official Apache-2.0 blocks.

M: widths base*[1,2,4,8,16], blocks [3,4,4,4,4,4,4,4,3],
expansion [2,3,4,4,4,4,4,3,2], kernel 3; base=32 by default.
Block checkpointing bounds activation memory without changing the architecture.
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .vendor.mednext_blocks import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock


class SingletonSafeGroupNorm(nn.GroupNorm):
    """Analytic GroupNorm for 1x1 toy inputs; ordinary sizes use PyTorch."""
    def forward(self, x):
        if x.shape[1] // self.num_groups * x.shape[-2] * x.shape[-1] == 1:
            return x * 0 + self.bias[None, :, None, None] + self.weight[None, :, None, None] * 0
        return super().forward(x)


def configure(block):
    old = block.norm
    block.norm = SingletonSafeGroupNorm(old.num_groups, old.num_channels, eps=old.eps)
    return block


def stage(channels, count, expansion):
    return nn.Sequential(*[configure(MedNeXtBlock(channels, channels, exp_r=expansion,
                          kernel_size=3, do_res=True, dim='2d')) for _ in range(count)])


def run_block(block, x):
    if block.training and torch.is_grad_enabled():
        return checkpoint(block, x, use_reentrant=False)
    return block(x)


def run_stage(blocks, x):
    for block in blocks:
        x = run_block(block, x)
    return x


class MedNeXtEncoder(nn.Module):
    def __init__(self, in_channels, base=32):
        super().__init__()
        widths = [base * 2**i for i in range(5)]
        self.stem = nn.Conv2d(in_channels, base, 1)
        self.stages = nn.ModuleList([stage(c, n, e) for c, n, e in
                                    zip(widths, (3,4,4,4,4), (2,3,4,4,4))])
        self.down = nn.ModuleList([configure(MedNeXtDownBlock(widths[i], widths[i+1],
                                 exp_r=(3,4,4,4)[i], kernel_size=3,
                                 do_res=True, dim='2d')) for i in range(4)])

    def forward(self, x):
        x = self.stem(x)
        features = []
        for i, blocks in enumerate(self.stages):
            x = run_stage(blocks, x)
            features.append(x)
            if i < 4:
                x = run_block(self.down[i], x)
        return tuple(features)


class MedNeXtDecoder(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        widths = [base*8, base*4, base*2, base]
        self.up = nn.ModuleList([configure(MedNeXtUpBlock(c*2, c, exp_r=e,
                               kernel_size=3, do_res=True, dim='2d'))
                               for c,e in zip(widths, (4,4,3,2))])
        self.stages = nn.ModuleList([stage(c,n,e) for c,n,e in
                                    zip(widths, (4,4,4,3), (4,4,3,2))])

    def decode_stage(self, i, x, skip):
        x = run_block(self.up[i], x)
        # Odd input sizes are supported; standard 256 inputs need no alignment.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return run_stage(self.stages[i], x + skip)

    def forward(self, features):
        x = features[-1]
        for i, skip in enumerate(reversed(features[:-1])):
            x = self.decode_stage(i, x, skip)
        return x
