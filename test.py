import argparse
import json
import os
from collections import defaultdict
from typing import Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
except ImportError as error:
    raise ImportError("nibabel is required for saving NIfTI. Please install via `pip install nibabel`.") from error

from models import ResUNetPP3DMultiHead
from utils import MultiSourceNiftiDataset3D, collect_cases, multiclass_dice_iou_hd95


def parse_num_classes(config: str) -> Dict[str, int]:
    if config:
        return json.loads(config)
    return {"2ch": 3, "4ch": 5, "sa": 4}


def resize_to_shape(label: torch.Tensor, shape):
    resized = F.interpolate(
        label.float().unsqueeze(0).unsqueeze(0),
        size=tuple(int(v) for v in shape),
        mode="nearest",
    )
    return resized.squeeze(0).squeeze(0).long()


def _normalize_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    slice_2d = slice_2d.astype(np.float32)
    min_val = float(slice_2d.min())
    max_val = float(slice_2d.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    normalized = (slice_2d - min_val) / (max_val - min_val)
    return (normalized * 255.0).clip(0, 255).astype(np.uint8)


def _label_to_rgb(label_2d: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 128, 255],
            [255, 255, 0],
            [255, 0, 255],
            [0, 255, 255],
            [255, 128, 0],
            [128, 0, 255],
            [0, 200, 120],
        ],
        dtype=np.uint8,
    )
    max_palette_index = palette.shape[0] - 1
    safe_label = np.clip(label_2d.astype(np.int64), 0, max_palette_index)
    return palette[safe_label]


def _overlay(image_gray_u8: np.ndarray, label_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    base = np.stack([image_gray_u8, image_gray_u8, image_gray_u8], axis=-1).astype(np.float32)
    color = _label_to_rgb(label_2d).astype(np.float32)
    mask = (label_2d > 0).astype(np.float32)[..., None]
    mixed = base * (1.0 - alpha * mask) + color * (alpha * mask)
    return mixed.clip(0, 255).astype(np.uint8)


def _choose_slice_index(mask_3d: np.ndarray) -> int:
    foreground_per_slice = (mask_3d > 0).reshape(mask_3d.shape[0], -1).sum(axis=1)
    if foreground_per_slice.max() <= 0:
        return int(mask_3d.shape[0] // 2)
    return int(np.argmax(foreground_per_slice))


def save_case_visualization(
    image_3d: np.ndarray,
    gt_3d: np.ndarray,
    pred_3d: np.ndarray,
    source: str,
    save_path: str,
) -> None:
    slice_idx = _choose_slice_index(gt_3d)

    image_slice = _normalize_to_uint8(image_3d[slice_idx])
    gt_slice = gt_3d[slice_idx]
    pred_slice = pred_3d[slice_idx]

    image_rgb = np.stack([image_slice, image_slice, image_slice], axis=-1)
    gt_rgb = _label_to_rgb(gt_slice)
    pred_rgb = _label_to_rgb(pred_slice)
    gt_overlay = _overlay(image_slice, gt_slice)
    pred_overlay = _overlay(image_slice, pred_slice)

    spacer = np.full((image_rgb.shape[0], 10, 3), 255, dtype=np.uint8)
    canvas = np.concatenate(
        [image_rgb, spacer, gt_rgb, spacer, pred_rgb, spacer, gt_overlay, spacer, pred_overlay],
        axis=1,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def main():
    parser = argparse.ArgumentParser(description="Test 3D ResUNet++ multi-head, evaluate metrics and export predictions")
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing sequence subdirectories")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save predicted NIfTIs")
    parser.add_argument("--source-order", nargs="+", default=["2ch", "4ch", "sa"], help="List of sequence sources")
    parser.add_argument("--image-dirname", type=str, default="image", help="Subdirectory name for images")
    parser.add_argument("--label-dirname", type=str, default="seg", help="Subdirectory name for labels")
    parser.add_argument(
        "--num-classes-json",
        type=str,
        default='{"2ch":3,"4ch":5,"sa":4}',
        help="JSON string specifying number of classes per source",
    )
    parser.add_argument("--input-size", nargs=3, type=int, default=[64, 160, 160], help="Target volume size (D, H, W)")
    parser.add_argument("--vis-per-source", type=int, default=3, help="Number of visualization slices per source")
    parser.add_argument("--vis-dir", type=str, default="", help="Custom visualization output directory")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation (e.g. cuda, cuda:0, cpu)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    num_classes_by_source = parse_num_classes(args.num_classes_json)

    all_cases = collect_cases(
        data_root=args.data_root,
        source_names=args.source_order,
        image_dirname=args.image_dirname,
        label_dirname=args.label_dirname,
    )
    print(f"Total test cases found: {len(all_cases)}")
    if len(all_cases) == 0:
        print("Error: No test cases found. Please check --data-root, --source-order, and directory names.")
        return

    dataset = MultiSourceNiftiDataset3D(
        cases=all_cases,
        output_size=args.input_size,
        source_num_classes=num_classes_by_source,
        augment=False,
    )

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = ResUNetPP3DMultiHead(
        in_channels=1,
        source_order=args.source_order,
        num_classes_by_source=num_classes_by_source,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    metrics = defaultdict(lambda: {"dice": [], "iou": [], "hd95": []})
    vis_count = defaultdict(int)
    vis_root = args.vis_dir if args.vis_dir else os.path.join(args.output_dir, "visualization")
    if args.vis_per_source > 0:
        os.makedirs(vis_root, exist_ok=True)

    with torch.no_grad():
        for item in dataset:
            image = item["image"].unsqueeze(0).to(device)
            mask = item["mask"].to(device)
            source = item["source"]
            source_id = int(item["source_id"].item())
            original_shape = tuple(int(v) for v in item["original_shape"].tolist())
            affine = item["affine"].numpy()

            logits = model.forward_source(image, source)
            pred = torch.argmax(logits, dim=1).squeeze(0)

            num_classes = num_classes_by_source[source]
            mean_dice, mean_iou, mean_hd95, _, _, _ = multiclass_dice_iou_hd95(pred, mask, num_classes)
            metrics[source]["dice"].append(mean_dice)
            metrics[source]["iou"].append(mean_iou)
            metrics[source]["hd95"].append(mean_hd95)

            pred_original = resize_to_shape(pred, original_shape).cpu().numpy().astype(np.uint8)

            base_name = os.path.basename(item["image_path"])
            save_name = f"src{source_id}_{source}_{base_name}"
            save_path = os.path.join(args.output_dir, save_name)
            nib.save(nib.Nifti1Image(pred_original, affine), save_path)

            if args.vis_per_source > 0 and vis_count[source] < args.vis_per_source:
                vis_name = os.path.splitext(os.path.splitext(base_name)[0])[0] + "_zview.png"
                vis_path = os.path.join(vis_root, source, vis_name)
                image_np = item["image"].squeeze(0).cpu().numpy()
                gt_np = mask.cpu().numpy().astype(np.int64)
                pred_np = pred.cpu().numpy().astype(np.int64)
                save_case_visualization(image_np, gt_np, pred_np, source, vis_path)
                vis_count[source] += 1

    print("\n--- Per-source evaluation results ---")
    for source in args.source_order:
        src_dice = metrics[source]["dice"]
        src_iou = metrics[source]["iou"]
        src_hd95 = metrics[source]["hd95"]
        mean_dice = float(np.mean(src_dice)) if src_dice else 0.0
        mean_iou = float(np.mean(src_iou)) if src_iou else 0.0
        mean_hd95 = float(np.mean(src_hd95)) if src_hd95 else 0.0
        print(f"  [{source}] Dice: {mean_dice:.4f} | IoU: {mean_iou:.4f} | HD95: {mean_hd95:.4f} | Samples: {len(src_dice)}")


if __name__ == "__main__":
    main()
