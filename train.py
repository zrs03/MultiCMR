import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models import ResUNetPP3DMultiHead
from utils import (
    MultiSourceNiftiDataset3D,
    collect_cases,
    dice_ce_loss,
    load_cases_from_yaml,
    multiclass_dice_iou,
    set_seed,
    split_cases_by_source,
)


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
    return {"2ch": 3, "4ch": 5, "sa": 4}


def get_model_core(model: torch.nn.Module) -> ResUNetPP3DMultiHead:
    if isinstance(model, DDP):
        return model.module
    return model


def run_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    source_order: list,
    num_classes_by_source: Dict[str, int],
    is_train: bool,
    is_distributed: bool = False,
    world_size: int = 1,
):
    if is_train:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    source_stats = defaultdict(lambda: {"dice": [], "iou": []})
    raw_model = get_model_core(model)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        source_id = batch["source_id"].to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        batch_loss = 0.0
        used_sources = 0

        with torch.set_grad_enabled(is_train):
            for src_index, src_name in enumerate(source_order):
                selected = source_id == src_index
                if not torch.any(selected):
                    continue

                src_images = images[selected]
                src_masks = masks[selected]
                logits = raw_model.forward_source(src_images, src_name)
                num_classes = num_classes_by_source[src_name]

                if logits.shape[2:] != src_masks.shape[1:]:
                    logits = F.interpolate(logits, size=src_masks.shape[1:], mode="trilinear", align_corners=False)

                loss = dice_ce_loss(logits, src_masks, num_classes)
                batch_loss = batch_loss + loss
                used_sources += 1

                dice, iou = multiclass_dice_iou(logits, src_masks, num_classes)
                source_stats[src_name]["dice"].append(float(dice.detach().cpu().item()))
                source_stats[src_name]["iou"].append(float(iou.detach().cpu().item()))

            if used_sources == 0:
                continue

            batch_loss = batch_loss / used_sources

            if is_train:
                batch_loss.backward()
                optimizer.step()

        loss_val = batch_loss.detach()
        if is_distributed:
            loss_val = reduce_tensor(loss_val, world_size)
        epoch_loss += float(loss_val.cpu().item())

    num_steps = max(len(loader), 1)
    epoch_loss = epoch_loss / num_steps

    summarized = {}
    for src in source_order:
        dice_values = source_stats[src]["dice"]
        iou_values = source_stats[src]["iou"]
        mean_dice = float(np.mean(dice_values)) if dice_values else 0.0
        mean_iou = float(np.mean(iou_values)) if iou_values else 0.0

        if is_distributed:
            d_tensor = torch.tensor(mean_dice, device=device)
            i_tensor = torch.tensor(mean_iou, device=device)
            d_tensor = reduce_tensor(d_tensor, world_size)
            i_tensor = reduce_tensor(i_tensor, world_size)
            mean_dice = float(d_tensor.cpu().item())
            mean_iou = float(i_tensor.cpu().item())

        summarized[src] = {"dice": mean_dice, "iou": mean_iou}

    return epoch_loss, summarized


def main():
    parser = argparse.ArgumentParser(description="Train 3D ResUNet++ multi-head on multi-sequence CMR NIfTI")
    parser.add_argument("--data-root", type=str, default="", help="Root folder containing sequence subdirectories")
    parser.add_argument("--split-yaml", type=str, default="preprocess/dataset_split.yaml", help="Path to preprocessed split YAML")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--source-order", nargs="+", default=None, help="List of sequence sources (defaults to YAML or [2ch 4ch sa])")
    parser.add_argument("--image-dirname", type=str, default="image", help="Subdirectory name for images")
    parser.add_argument("--label-dirname", type=str, default="seg", help="Subdirectory name for segmentation labels")
    parser.add_argument(
        "--num-classes-json",
        type=str,
        default=None,
        help="JSON string specifying number of classes per source (defaults to YAML)",
    )
    parser.add_argument("--input-size", nargs=3, type=int, default=[64, 160, 160], help="Target volume size (D, H, W)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio if split-yaml is not used")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for single GPU training (e.g. cuda, cuda:0, cpu)",
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

    # Case 1: Load from Split YAML
    if args.split_yaml and os.path.exists(args.split_yaml):
        if is_master_proc(rank):
            print(f"[MultiCMR] Loading dataset split from YAML: {args.split_yaml}")
        train_cases, info = load_cases_from_yaml(args.split_yaml, split="train", data_root_override=args.data_root or None)
        val_cases, _ = load_cases_from_yaml(args.split_yaml, split="val", data_root_override=args.data_root or None)
        source_order = args.source_order or info["source_order"]
        num_classes_by_source = parse_num_classes(args.num_classes_json) if args.num_classes_json else info["num_classes_by_source"]
    else:
        # Case 2: Direct scan from data_root
        if not args.data_root:
            raise ValueError(f"Neither valid --split-yaml nor --data-root was provided! Missing: {args.split_yaml}")
        source_order = args.source_order or ["2ch", "4ch", "sa"]
        num_classes_by_source = parse_num_classes(args.num_classes_json)
        all_cases = collect_cases(
            data_root=args.data_root,
            source_names=source_order,
            image_dirname=args.image_dirname,
            label_dirname=args.label_dirname,
        )
        train_cases, val_cases = split_cases_by_source(
            all_cases,
            source_names=source_order,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    if is_master_proc(rank):
        print(f"[MultiCMR] Cases: Train={len(train_cases)}, Val={len(val_cases)}")
        print(f"[MultiCMR] Source Order: {source_order}")
        print(f"[MultiCMR] Num Classes: {num_classes_by_source}")

    train_dataset = MultiSourceNiftiDataset3D(
        cases=train_cases,
        output_size=args.input_size,
        source_num_classes=num_classes_by_source,
        augment=True,
    )
    val_dataset = MultiSourceNiftiDataset3D(
        cases=val_cases,
        output_size=args.input_size,
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

    model = ResUNetPP3DMultiHead(
        in_channels=1,
        source_order=source_order,
        num_classes_by_source=num_classes_by_source,
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
            file.write(f"3D Multi-Head CMR Training Log (World size: {world_size})\n")

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
                        "input_size": args.input_size,
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
