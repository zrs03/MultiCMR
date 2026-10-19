import argparse
import json
import os
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from models import MMFormerAnatomy
from utils.sampling import PatientWindowSampler
from utils.optimization import build_adamw
from utils.losses import masked_segmentation_loss
from utils.checkpoint import ARCHITECTURE, atomic_save, capture_rng_state, restore_training_state
from preprocess.processing import validate_output_sizes
from utils import (
    MultiSourceNiftiDataset3D,
    load_cases_from_yaml,
    multiclass_dice_iou,
    set_seed,
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
    aux_weight=.2, is_distributed=False, world_size=1, fp16=False, scaler=None, deep_weight=.1,
):
    if fp16 and torch.device(device).type != "cuda":
        raise ValueError("--fp16 requires a CUDA device")
    if is_train and fp16 and (scaler is None or not scaler.is_enabled()):
        raise ValueError("FP16 training requires an enabled GradScaler")
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
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16, enabled=fp16):
                outputs = model(images, modality_mask)
            batch_loss = masked_segmentation_loss(
                outputs, targets, num_classes_by_source, aux_weight=aux_weight if is_train else 0.,
                deep_weight=deep_weight if is_train else 0.,
                boundary_weight=boundary_weight, boundary_downsample=boundary_downsample,
                boundary_max_distance=boundary_max_distance)
            with torch.no_grad():
                for i, view in enumerate(source_order):
                    valid = modality_mask[:, i].bool()
                    n = int(valid.sum())
                    if not n:
                        continue
                    logits = outputs['logits'][view][valid].float()
                    dice, iou = multiclass_dice_iou(logits, targets[view][valid], num_classes_by_source[view])
                    stats[view] += torch.stack((dice * n, iou * n, logits.new_tensor(n)))
            if is_train:
                if fp16:
                    scaler.scale(batch_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
    parser = argparse.ArgumentParser(description="Train center-window LAX + 3D SAX with masked token fusion")
    parser.add_argument("--split-yaml", type=str, default="preprocess/matched.yaml", help="Path to preprocessed split YAML")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Directory to save checkpoints and logs")
    parser.add_argument("--source-order", nargs="+", default=None, help="List of sequence sources (defaults to YAML or [2ch 4ch sa])")
    parser.add_argument(
        "--num-classes-json",
        type=str,
        default=None,
        help="JSON string specifying number of classes per source (defaults to YAML)",
    )
    parser.add_argument('--lax-size', nargs=3, type=int, default=None, help='2CH/4CH K H W (default 5 128 128 or YAML)')
    parser.add_argument('--sax-size', nargs=3, type=int, default=None, help='SAX D H W (default 128 128 128 or YAML)')
    parser.add_argument('--base', type=int, default=16)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--token-grid', nargs=3, type=int, default=(2, 4, 4))
    parser.add_argument('--intra-layers', type=int, default=1)
    parser.add_argument('--token-drop-rate', type=float, default=.05,
                        help='Training anatomy-read token masking: default 0.05 for fresh and resumed training; 0 disables')
    parser.add_argument('--inter-layers', type=int, default=2, help='Must be 2: LAX/anatomy cross-attention, then SAX/anatomy cross-attention')
    parser.add_argument('--deep-weight', type=float, default=.1, help='Single-scale d2 Dice+CE weight; 0 disables')
    parser.add_argument('--aux-weight', type=float, default=.2)
    parser.add_argument('--windows-per-patient', type=int, default=4, help='Max distinct windows per patient per epoch; 0 uses all')
    parser.add_argument('--affine-prob', type=float, default=.5)
    parser.add_argument('--rotation-degrees', type=float, default=10.)
    parser.add_argument('--translation-fraction', type=float, default=.05)
    parser.add_argument('--scale-fraction', type=float, default=.1)
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume; --epochs is the total target epoch")
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW decay for convolution/linear weights")
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
    parser.add_argument("--fp16", action="store_true", help="Enable CUDA FP16 automatic mixed precision")
    args = parser.parse_args()
    if not 0 <= args.deep_weight < float('inf'):
        parser.error('--deep-weight must be finite and nonnegative')
    if args.epochs < 1:
        parser.error("--epochs must be positive")

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
        print(f"[MultiCMR] Window samples: Train={len(train_cases)}, Val={len(val_cases)}")
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
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resume_checkpoint.get('architecture') != ARCHITECTURE:
            raise ValueError('Resume requires a joint-LAX patch attention / SAX anatomy cross-attention patch-size-2 checkpoint (v13, single-scale deep supervision); older checkpoints have incompatible model/optimizer parameters')
        if 'optimizer' in resume_checkpoint and resume_checkpoint.get('optimizer_type') != 'AdamW':
            raise ValueError('Resume checkpoint optimizer is not AdamW; full optimizer restoration is incompatible')
        if list(resume_checkpoint['source_order']) != list(source_order):
            raise ValueError('Resume checkpoint source order differs from current data')
        if resume_checkpoint['num_classes_by_source'] != num_classes_by_source:
            raise ValueError('Resume checkpoint class counts differ from current data')
        if validate_output_sizes(resume_checkpoint['output_sizes']) != output_sizes:
            raise ValueError('Resume checkpoint window/spatial sizes differ from current preprocessing')
        # The checkpoint owns architecture settings; no need to repeat --base, --dim, etc.
        model_config = dict(resume_checkpoint['model_config'])
        if model_config['temporal_size'] != output_sizes['2ch'][0]:
            raise ValueError('Resume checkpoint temporal_size is inconsistent with preprocessing')
    # Training regularization follows this invocation, including on resume.
    model_config['token_drop_rate'] = args.token_drop_rate
    if not 0 <= model_config['token_drop_rate'] < 1:
        parser.error('--token-drop-rate must be in [0, 1)')
    if is_master_proc(rank):
        print(f'[MultiCMR] Single-scale d2 supervision weight: {args.deep_weight}; training only')
        print(f"[MultiCMR] Anatomy-read token masking: rate={model_config['token_drop_rate']}; train only, >=2 present views")
        print(f'[MultiCMR] Per-view sizes: {output_sizes}; train/val=center-window + shared SAX')

    train_dataset = MultiSourceNiftiDataset3D(
        cases=train_cases,
        output_sizes=output_sizes,
        source_num_classes=num_classes_by_source,
        augment=True,
        affine_prob=args.affine_prob, rotation_degrees=args.rotation_degrees,
        translation_fraction=args.translation_fraction, scale_fraction=args.scale_fraction,
    )
    val_dataset = MultiSourceNiftiDataset3D(
        cases=val_cases,
        output_sizes=output_sizes,
        source_num_classes=num_classes_by_source,
        augment=False,
    )

    train_sampler = PatientWindowSampler(train_cases, args.windows_per_patient, args.seed,
                                         rank=rank, num_replicas=world_size)
    val_sampler = (DistributedSampler(val_dataset, shuffle=False, drop_last=False)
                   if is_distributed else None)
    if is_master_proc(rank):
        print(f'[MultiCMR] Patients={len(train_sampler.groups)}; selected windows/epoch='
              f'{train_sampler.selected_count}; DDP padded total={train_sampler.total_size}; '
              f'windows/rank={len(train_sampler)}; steps/rank={(len(train_sampler) + args.batch_size - 1) // args.batch_size}')
        print(f'[MultiCMR] Spatial affine: probability={args.affine_prob}, rotation=±{args.rotation_degrees}deg, '
              f'translation=±{args.translation_fraction}, scale=1±{args.scale_fraction}; validation=all windows, no augmentation')

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

    if args.fp16 and device.type != "cuda":
        raise ValueError("--fp16 requires a CUDA device")
    # torch.amp.GradScaler is unavailable in older supported PyTorch 2.x releases.
    scaler = (torch.amp.GradScaler("cuda", enabled=args.fp16)
              if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=args.fp16))
    optimizer = build_adamw(get_model_core(model), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5)

    start_epoch, best_val_dice = 1, -1.0
    if resume_checkpoint is not None:
        start_epoch, best_val_dice = restore_training_state(
            resume_checkpoint, get_model_core(model), optimizer, scheduler, rank, world_size, scaler=scaler)
        del resume_checkpoint
    last_ckpt = os.path.join(args.output_dir, "last_model_3d_multihead.pth")
    if is_master_proc(rank):
        print(f"[MultiCMR] Precision: {'FP16 AMP' if args.fp16 else 'FP32'}; losses/metrics use FP32")
        print(f"[MultiCMR] AdamW: lr={optimizer.param_groups[0]['lr']:.8g}; "
              f"weight_decay groups={[g['weight_decay'] for g in optimizer.param_groups]}")
    best_ckpt = os.path.join(args.output_dir, "best_model_3d_multihead.pth")
    log_path = os.path.join(args.output_dir, "train_log.txt")

    if is_master_proc(rank):
        with open(log_path, "a" if args.resume else "w", encoding="utf-8") as file:
            file.write(f"Center-window LAX + 3D SAX Training Log (World size: {world_size})\n")
            if args.resume:
                message = (f"Resume: {args.resume} | next_epoch={start_epoch} | "
                           f"best_val_dice={best_val_dice:.6f} | lr={optimizer.param_groups[0]['lr']:.8g}")
                print(message)
                file.write(message + "\n")
        if start_epoch > args.epochs:
            print(f'Checkpoint already reached target epoch {args.epochs}; no training steps needed.')

    for epoch in range(start_epoch, args.epochs + 1):
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
            aux_weight=args.aux_weight, deep_weight=args.deep_weight,
            boundary_weight=args.boundary_weight,
            boundary_downsample=args.boundary_downsample,
            boundary_max_distance=args.boundary_max_distance,
            is_distributed=is_distributed,
            world_size=world_size, fp16=args.fp16, scaler=scaler,
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
            world_size=world_size, fp16=args.fp16, scaler=scaler,
        )

        scheduler.step(val_loss)

        val_dice_values = [val_stats[src]["dice"] for src in source_order]
        mean_val_dice = float(np.mean(val_dice_values)) if val_dice_values else 0.0

        improved = mean_val_dice > best_val_dice
        best_val_dice = max(best_val_dice, mean_val_dice)
        local_rng = capture_rng_state()
        if is_distributed:
            rng_states = [None] * world_size
            dist.all_gather_object(rng_states, local_rng)
        else:
            rng_states = [local_rng]
        if is_master_proc(rank):
            checkpoint = {
                "model": get_model_core(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "optimizer_type": "AdamW",
                "scaler": scaler.state_dict(),
                "precision": "fp16" if args.fp16 else "fp32",
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_dice": best_val_dice,
                "rng_states": rng_states,
                "world_size": world_size,
                "train_args": vars(args),
                "source_order": source_order,
                "num_classes_by_source": num_classes_by_source,
                "output_sizes": output_sizes,
                "model_config": model_config,
                "architecture": ARCHITECTURE,
            }
            atomic_save(checkpoint, last_ckpt)
            if improved:
                atomic_save(checkpoint, best_ckpt)

            elapsed = time.time() - epoch_start
            line = (
                f"Epoch {epoch:03d} | {elapsed:.1f}s | train_steps={len(train_loader)} | "
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
        print(f"Training completed. Latest checkpoint: {last_ckpt}; historical best Dice={best_val_dice:.6f}")

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
