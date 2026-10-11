import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models import MMFormerAnatomy
from preprocess.processing import validate_output_sizes
from utils import (
    MultiSourceNiftiDataset3D,
    dice_ce_loss_per_sample,
    load_cases_from_yaml,
    multiclass_dice_iou,
    set_seed,
)


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

def init_distributed() -> Tuple[bool, int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
        else:
            backend = "gloo"

        dist.init_process_group(backend=backend, init_method="env://")
        return True, rank, local_rank, world_size
    return False, 0, 0, 1


def is_master_proc(rank: int) -> bool:
    return rank == 0


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def parse_num_classes(config: str) -> Dict[str, int]:
    if config:
        return json.loads(config)
    return {"2ch": 3, "4ch": 6, "sa": 4}


def sample_modality_mask(
    batch_size: int,
    num_modalities: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Randomly sample a non-empty multi-hot modality mask.
    """
    if num_modalities != 3:
        raise ValueError(
            "Current modality-mask sampling expects exactly "
            "3 modalities: 2ch / 4ch / sa."
        )

    candidates = torch.tensor(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=torch.float32,
        device=device,
    )

    indices = torch.randint(
        low=0,
        high=len(candidates),
        size=(batch_size,),
        device=device,
    )

    return candidates[indices]


def get_model_core(model: torch.nn.Module):
    if isinstance(model, DDP):
        return model.module
    return model


def run_one_epoch(
    model, loader, optimizer, device, source_order, num_classes_by_source,
    is_train, boundary_weight=.05, boundary_downsample=2, boundary_max_distance=10.,
    aux_weight=.2, deep_weight=.1, is_distributed=False, world_size=1,
):
    model.train(is_train)
    epoch_loss, steps = 0., 0
    # Sums and counts exclude absent views from both loss and metrics.
    stats = {v: torch.zeros(3, device=device) for v in source_order}
    for batch in loader:
        images = {v: batch['image'][v].to(device, non_blocking=True) for v in source_order}
        if not batch['masks']:
            raise RuntimeError('Training/validation requires annotations')
        targets = {v: batch['masks'][v].to(device, non_blocking=True) for v in source_order}
        batch_size = images['sa'].shape[0]
        modality_mask = (sample_modality_mask(batch_size, len(source_order), device)
                         if is_train else torch.ones(batch_size, len(source_order), device=device))
        # The model selects present samples before encoding, for both 2D and 3D inputs.
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            outputs = model(images, modality_mask)
            loss_sum, count = 0., 0
            for i, view in enumerate(source_order):
                valid = modality_mask[:, i].bool()
                n = int(valid.sum())
                if not n:
                    continue
                logits = outputs['logits'][view][valid]
                target = targets[view][valid]
                classes = num_classes_by_source[view]
                loss = dice_ce_loss_per_sample(logits, target, classes, framewise=view != 'sa')
                if boundary_weight > 0:
                    loss = loss + boundary_weight * boundary_loss_per_sample(
                        logits, target, classes, boundary_downsample, boundary_max_distance, framewise=view != 'sa')
                if is_train and aux_weight > 0:
                    loss = loss + aux_weight * dice_ce_loss_per_sample(
                        outputs['aux_logits'][view][valid], target, classes, framewise=view != 'sa')
                if is_train and deep_weight > 0:
                    for pred in outputs['deep_logits'][view]:
                        loss = loss + deep_weight * dice_ce_loss_per_sample(pred[valid], target, classes, framewise=view != 'sa')
                loss_sum = loss_sum + loss.sum()
                count += n
                with torch.no_grad():
                    dice, iou = multiclass_dice_iou(logits, target, classes, framewise=view != 'sa')
                    stats[view] += torch.stack((dice * n, iou * n, logits.new_tensor(n)))
            batch_loss = loss_sum / count
            if is_train:
                batch_loss.backward()
                optimizer.step()
        epoch_loss += float(batch_loss.detach())
        steps += 1
    totals = torch.tensor([epoch_loss, steps], device=device, dtype=torch.float64)
    if is_distributed:
        dist.all_reduce(totals)
        for values in stats.values():
            dist.all_reduce(values)
    summarized = {v: {'dice': float(values[0] / values[2].clamp_min(1)),
                      'iou': float(values[1] / values[2].clamp_min(1))}
                  for v, values in stats.items()}
    return float(totals[0] / totals[1].clamp_min(1)), summarized


def main():
    parser = argparse.ArgumentParser(description="Train full-cine 2D+t LAX + 3D SAX with masked token fusion")
    parser.add_argument("--split-yaml", type=str, default="preprocess/matched.yaml", help="Path to preprocessed split YAML")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--source-order", nargs="+", default=None, help="List of sequence sources (defaults to YAML or [2ch 4ch sa])")
    parser.add_argument(
        "--num-classes-json",
        type=str,
        default=None,
        help="JSON string specifying number of classes per source (defaults to YAML)",
    )
    parser.add_argument('--lax-size', nargs=3, type=int, default=None, help='2CH/4CH T H W (default 64 128 128 or YAML)')
    parser.add_argument('--sax-size', nargs=3, type=int, default=None, help='SAX D H W (default 128 128 128 or YAML)')
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--token-grid', nargs=3, type=int, default=(2, 4, 4))
    parser.add_argument('--intra-layers', type=int, default=1)
    parser.add_argument('--inter-layers', type=int, default=2)
    parser.add_argument('--aux-weight', type=float, default=.2)
    parser.add_argument('--deep-weight', type=float, default=.1)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for single GPU training (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--boundary-weight", type=float, default=0.05,
        help="Weight of signed-distance boundary loss (0 disables it).",
    )
    parser.add_argument(
        "--boundary-downsample", type=int, default=2,
        help="Compute distance loss at 1/N spatial scale (1=full resolution).",
    )
    parser.add_argument(
        "--boundary-max-distance", type=float, default=10.0,
        help="Clip signed distance to +/- this many voxels at loss scale.",
    )
    args = parser.parse_args()

    is_distributed, rank, local_rank, world_size = init_distributed()
    set_seed(args.seed + rank)

    if is_distributed:
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if is_master_proc(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[MultiCMR] Running in {'Distributed (DDP, World Size=' + str(world_size) + ')' if is_distributed else 'Single-GPU/CPU'} mode")
        print(f"[MultiCMR] Master rank using device: {device}")

    # ============================================================
    # Load train / val cases from Split YAML
    # ============================================================

    if not os.path.exists(args.split_yaml):
        raise FileNotFoundError(
            f"Split YAML not found: {args.split_yaml}"
        )

    if is_master_proc(rank):
        print(
            f"[MultiCMR] Loading dataset split from YAML: "
            f"{args.split_yaml}"
        )

    train_cases, info = load_cases_from_yaml(
        args.split_yaml,
        split="train",
    )

    val_cases, _ = load_cases_from_yaml(
        args.split_yaml,
        split="val",
    )

    source_order = (
        args.source_order
        or info["source_order"]
    )

    num_classes_by_source = (
        parse_num_classes(args.num_classes_json)
        if args.num_classes_json
        else info["num_classes_by_source"]
    )

    if is_master_proc(rank):
        print(f"[MultiCMR] Cases: Train={len(train_cases)}, Val={len(val_cases)}")
        print(f"[MultiCMR] Source Order: {source_order}")
        print(f"[MultiCMR] Num Classes: {num_classes_by_source}")

    output_sizes = validate_output_sizes(info.get('output_sizes'))
    if args.lax_size:
        output_sizes.update({'2ch': args.lax_size, '4ch': args.lax_size})
    if args.sax_size:
        output_sizes['sa'] = args.sax_size
    output_sizes = validate_output_sizes(output_sizes)
    model_config = dict(base=args.base, dim=args.dim, heads=args.heads,
                        token_grid=args.token_grid, intra_layers=args.intra_layers,
                        inter_layers=args.inter_layers,
                        temporal_size=max(output_sizes[v][0] for v in ('2ch', '4ch')))
    if is_master_proc(rank):
        print(f'[MultiCMR] Per-view sizes: {output_sizes}; train/val=full 2D+t cine')

    train_dataset = MultiSourceNiftiDataset3D(
        cases=train_cases,
        output_sizes=output_sizes,
        source_num_classes=num_classes_by_source,
        augment=True,
    )
    val_dataset = MultiSourceNiftiDataset3D(
        cases=val_cases,
        output_sizes=output_sizes,
        source_num_classes=num_classes_by_source,
        augment=False,
    )

    if is_distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, drop_last=False)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = MMFormerAnatomy(
        in_channels=1,
        source_order=source_order,
        num_classes_by_source=num_classes_by_source,
        **model_config,
    ).to(device)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
            output_device=local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    best_val_dice = -1.0
    best_ckpt = os.path.join(args.output_dir, "best_model_3d_multihead.pth")
    log_path = os.path.join(args.output_dir, "train_log.txt")

    if is_master_proc(rank):
        with open(log_path, "w", encoding="utf-8") as file:
            file.write(f"2D+t LAX + 3D SAX Training Log (World size: {world_size})\n")

    for epoch in range(1, args.epochs + 1):
        if is_distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        epoch_start = time.time()

        train_loss, train_stats = run_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            source_order=source_order,
            num_classes_by_source=num_classes_by_source,
            is_train=True,
            aux_weight=args.aux_weight,
            deep_weight=args.deep_weight,
            boundary_weight=args.boundary_weight,
            boundary_downsample=args.boundary_downsample,
            boundary_max_distance=args.boundary_max_distance,
            is_distributed=is_distributed,
            world_size=world_size,
        )

        val_loss, val_stats = run_one_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            source_order=source_order,
            num_classes_by_source=num_classes_by_source,
            is_train=False,
            boundary_weight=args.boundary_weight,
            boundary_downsample=args.boundary_downsample,
            boundary_max_distance=args.boundary_max_distance,
            is_distributed=is_distributed,
            world_size=world_size,
        )

        scheduler.step(val_loss)

        val_dice_values = [val_stats[src]["dice"] for src in source_order]
        mean_val_dice = float(np.mean(val_dice_values)) if val_dice_values else 0.0

        if is_master_proc(rank):
            if mean_val_dice > best_val_dice:
                best_val_dice = mean_val_dice
                raw_model = get_model_core(model)
                torch.save(
                    {
                        "model": raw_model.state_dict(),
                        "source_order": source_order,
                        "num_classes_by_source": num_classes_by_source,
                        "output_sizes": output_sizes,
                        "model_config": model_config,
                        "architecture": "temporal_2d_lax_3d_sax_v1",
                    },
                    best_ckpt,
                )

            elapsed = time.time() - epoch_start
            line = (
                f"Epoch {epoch:03d} | {elapsed:.1f}s | "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} mean_val_dice={mean_val_dice:.4f}\n"
            )
            for src in source_order:
                line += (
                    f"  {src}: train_dice={train_stats[src]['dice']:.4f}, train_iou={train_stats[src]['iou']:.4f}, "
                    f"val_dice={val_stats[src]['dice']:.4f}, val_iou={val_stats[src]['iou']:.4f}\n"
                )

            print(line, end="")
            with open(log_path, "a", encoding="utf-8") as file:
                file.write(line)

    if is_master_proc(rank):
        print(f"Training completed. Best checkpoint saved at: {best_ckpt}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
