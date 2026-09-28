import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

def dice_ce_loss_per_sample(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Calculate Dice + Cross-Entropy loss for each sample.

    Args:
        logits:
            [B, C, D, H, W]

        target:
            [B, D, H, W]

        num_classes:
            Number of segmentation classes.

    Returns:
        loss:
            [B]

    Notes:
        The returned loss is per-sample so that the caller can
        apply sample-specific weights, e.g. modality-missing
        temperature weights.
    """

    if logits.ndim != 5:
        raise ValueError(
            f"logits must be [B,C,D,H,W], "
            f"got {logits.shape}"
        )

    if target.ndim != 4:
        raise ValueError(
            f"target must be [B,D,H,W], "
            f"got {target.shape}"
        )

    batch_size = logits.shape[0]

    # ============================================================
    # 1. Cross Entropy
    # ============================================================

    ce_loss = F.cross_entropy(
        logits,
        target,
        reduction="none",
    )

    # [B,D,H,W]
    ce_loss = ce_loss.reshape(
        batch_size,
        -1,
    )

    # [B]
    ce_loss = ce_loss.mean(dim=1)

    # ============================================================
    # 2. Dice
    # ============================================================

    probs = torch.softmax(
        logits,
        dim=1,
    )

    target_one_hot = F.one_hot(
        target,
        num_classes=num_classes,
    )

    # [B,D,H,W,C]
    target_one_hot = target_one_hot.permute(
        0,
        4,
        1,
        2,
        3,
    ).float()

    # ------------------------------------------------------------
    # Flatten spatial dimensions
    # ------------------------------------------------------------

    probs = probs.reshape(
        batch_size,
        num_classes,
        -1,
    )

    target_one_hot = target_one_hot.reshape(
        batch_size,
        num_classes,
        -1,
    )

    # ------------------------------------------------------------
    # Per-class Dice
    # ------------------------------------------------------------

    intersection = (
        probs * target_one_hot
    ).sum(dim=2)

    denominator = (
        probs.sum(dim=2)
        + target_one_hot.sum(dim=2)
    )

    dice = (
        2.0 * intersection + smooth
    ) / (
        denominator + smooth
    )

    # [B]
    dice = dice.mean(dim=1)

    dice_loss = 1.0 - dice

    # ============================================================
    # 3. Dice + CE
    # ============================================================

    loss = dice_loss + ce_loss

    return loss


def dice_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    smooth: float = 1e-5,
) -> torch.Tensor:
    """
    Standard scalar Dice + CE loss.

    Kept for compatibility with other code.
    """

    loss = dice_ce_loss_per_sample(
        logits=logits,
        target=target,
        num_classes=num_classes,
        smooth=smooth,
    )

    return loss.mean()


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_ce_loss(logits, target, self.num_classes)


def multiclass_dice_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int):
    pred = torch.argmax(logits, dim=1)
    eps = 1e-6
    dice_scores = []
    iou_scores = []

    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()

        intersection = (pred_mask * target_mask).sum(dim=(1, 2, 3))
        pred_sum = pred_mask.sum(dim=(1, 2, 3))
        target_sum = target_mask.sum(dim=(1, 2, 3))

        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        union = pred_sum + target_sum - intersection
        iou = (intersection + eps) / (union + eps)

        dice_scores.append(dice.mean())
        iou_scores.append(iou.mean())

    if not dice_scores:
        zero_tensor = torch.tensor(0.0, device=logits.device)
        return zero_tensor, zero_tensor

    return torch.stack(dice_scores).mean(), torch.stack(iou_scores).mean()


def surface_distances(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)

    if not mask_a.any() and not mask_b.any():
        return np.asarray([0.0], dtype=np.float32)
    if not mask_a.any() or not mask_b.any():
        max_dist = float(np.linalg.norm(mask_a.shape))
        return np.asarray([max_dist], dtype=np.float32)

    structure = generate_binary_structure(mask_a.ndim, 1)
    surface_a = np.logical_xor(mask_a, binary_erosion(mask_a, structure=structure, border_value=0))
    surface_b = np.logical_xor(mask_b, binary_erosion(mask_b, structure=structure, border_value=0))

    if not surface_a.any() or not surface_b.any():
        max_dist = float(np.linalg.norm(mask_a.shape))
        return np.asarray([max_dist], dtype=np.float32)

    dt_b = distance_transform_edt(~surface_b)
    dt_a = distance_transform_edt(~surface_a)

    dist_a_to_b = dt_b[surface_a]
    dist_b_to_a = dt_a[surface_b]
    return np.concatenate([dist_a_to_b, dist_b_to_a]).astype(np.float32)


def hd95_binary(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    distances = surface_distances(pred_mask, target_mask)
    return float(np.percentile(distances, 95))


def multiclass_dice_iou_hd95(pred: torch.Tensor, target: torch.Tensor, num_classes: int):
    eps = 1e-6
    dice_scores = []
    iou_scores = []
    hd95_scores = []

    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    for class_id in range(1, num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        pred_sum = pred_mask.sum()
        target_sum = target_mask.sum()

        dice = (2.0 * intersection + eps) / (pred_sum + target_sum + eps)
        union = pred_sum + target_sum - intersection
        iou = (intersection + eps) / (union + eps)

        dice_scores.append(float(dice.item()))
        iou_scores.append(float(iou.item()))

        pred_mask_np = pred_np == class_id
        target_mask_np = target_np == class_id
        hd95_scores.append(hd95_binary(pred_mask_np, target_mask_np))

    if not dice_scores:
        return 0.0, 0.0, 0.0, [], [], []

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    mean_hd95 = float(np.mean(hd95_scores))

    return mean_dice, mean_iou, mean_hd95, dice_scores, iou_scores, hd95_scores
