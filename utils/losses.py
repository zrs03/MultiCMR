"""Training losses: Dice+CE, signed-distance boundary, and masked auxiliary supervision.

All probability and reduction math uses FP32, including under FP16 AMP.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

def dice_ce_loss_per_sample(logits, target, num_classes, smooth=1e-5, framewise=False):
    """FP32 Dice (including background) + unweighted CE, returned as [B]."""
    logits = logits.float()
    if framewise:
        batch, classes, time, height, width = logits.shape
        frames = logits.permute(0, 2, 1, 3, 4).reshape(batch * time, classes, height, width)
        return dice_ce_loss_per_sample(
            frames, target.reshape(batch * time, height, width), num_classes, smooth
        ).reshape(batch, time).mean(1)
    if logits.ndim not in (4, 5) or target.ndim != logits.ndim - 1:
        raise ValueError('Expected 2D/3D logits [B,C,...] and target [B,...]')
    if logits.shape[0] != target.shape[0] or logits.shape[1] != num_classes:
        raise ValueError('Logits/target/num_classes mismatch')
    ce = F.cross_entropy(logits, target, reduction='none').flatten(1).mean(1)
    probs = logits.softmax(1).flatten(2)
    one_hot = F.one_hot(target, num_classes).movedim(-1, 1).float().flatten(2)
    intersection = (probs * one_hot).sum(2)
    dice = (2 * intersection + smooth) / (probs.sum(2) + one_hot.sum(2) + smooth)
    return ce + 1 - dice.mean(1)


def boundary_loss_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    downsample: int = 2,
    max_distance: float = 10.0,
    framewise: bool = False,
) -> torch.Tensor:
    """Signed-distance boundary loss (foreground classes only), shape [B].
    A true GT signed distance field is computed without gradient using SciPy.
    Output probabilities remain differentiable. Inside distances are negative.
    Empty foreground classes are skipped; foreground classes are averaged.
    downsample=2 computes distance maps on a half-resolution grid for speed.
    For very thin structures, use --boundary-downsample 1.
    """
    logits = logits.float()
    if framewise:
        batch, classes, time, height, width = logits.shape
        frames = logits.permute(0, 2, 1, 3, 4).reshape(batch * time, classes, height, width)
        values = boundary_loss_per_sample(frames, target.reshape(batch * time, height, width),
                                          num_classes, downsample, max_distance)
        return values.reshape(batch, time).mean(1)
    if logits.ndim not in (4, 5) or target.ndim != logits.ndim - 1:
        raise ValueError("Expected 2D/3D logits [B,C,...] and target [B,...]")
    if logits.shape[0] != target.shape[0] or logits.shape[1] != num_classes:
        raise ValueError("Logits/target/num_classes mismatch")
    if downsample < 1 or max_distance <= 0:
        raise ValueError("downsample must be >=1; max_distance must be positive")
    if downsample > 1:
        size = tuple(max(1, (d + downsample - 1) // downsample) for d in target.shape[1:])
        # Resize probability maps and labels to the SAME coarse grid.
        probs = F.interpolate(torch.softmax(logits.float(), dim=1), size=size,
                              mode="bilinear" if logits.ndim == 4 else "trilinear", align_corners=False)
        gt = F.interpolate(target[:, None].float(), size=size,
                           mode="nearest")[:, 0].long()
    else:
        probs = torch.softmax(logits.float(), dim=1)
        gt = target.long()
    gt_np = gt.detach().cpu().numpy()
    losses = []
    for b in range(gt.shape[0]):
        per_class = []
        for c in range(1, num_classes):  # ignore background
            mask = (gt_np[b] == c)
            if not mask.any() or mask.all():
                continue
            inside = distance_transform_edt(mask)
            outside = distance_transform_edt(~mask)
            signed = np.clip(outside - inside, -max_distance, max_distance)
            signed = np.asarray(signed / max_distance, dtype=np.float32)
            signed_tensor = torch.from_numpy(signed).to(device=probs.device)
            per_class.append((probs[b, c] * signed_tensor).mean())
        # Return a graph-connected zero for samples without foreground.
        losses.append(torch.stack(per_class).mean() if per_class
                      else probs[b].sum() * 0.0)
    return torch.stack(losses)


def masked_segmentation_loss(result, targets, num_classes_by_source=None,
                             aux_weight=.2, boundary_weight=.05,
                             boundary_downsample=2, boundary_max_distance=10.):
    """Average over present sample/view pairs; never supervise missing views.

    Main Dice includes background. Boundary averages nonempty foreground classes.
    Auxiliary logits are produced only during training. No deep supervision.
    """
    total, count = 0., 0
    for i, view in enumerate(('2ch', '4ch', 'sa')):
        logits_all = result['logits'][view]
        valid = result['valid_views'][:, i].bool()
        n = int(valid.sum())
        if not n:
            continue
        logits = logits_all[valid].float()
        target = targets[view][valid].long()
        classes = logits.shape[1] if num_classes_by_source is None else num_classes_by_source[view]
        loss = dice_ce_loss_per_sample(logits, target, classes)
        if boundary_weight > 0:
            loss = loss + boundary_weight * boundary_loss_per_sample(
                logits, target, classes, boundary_downsample, boundary_max_distance)
        if aux_weight > 0 and view in result.get('aux_logits', {}):
            loss = loss + aux_weight * dice_ce_loss_per_sample(
                result['aux_logits'][view][valid].float(), target, classes)
        total = total + loss.sum()
        count += n
    if count == 0:
        raise ValueError('No valid supervised views')
    return total / count
