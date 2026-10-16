"""Evaluate the MMFormerAnatomy training checkpoint on a YAML split.

Compatible with train.py's load_cases_from_yaml / MultiSourceNiftiDataset3D
and MMFormerAnatomy(x, modality_mask) interface.

Example:
  python test.py --split-yaml preprocess/matched.yaml \
    --checkpoint checkpoints/best_model_3d_multihead.pth --split test

LAX enters as centered temporal windows; outputs are reconstructed to native frame counts. Exports use model space: LAX [H,W,1,T] and SAX [H,W,D].
"""
from __future__ import annotations
import argparse
import json
import csv
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError('Install nibabel: pip install nibabel') from exc
try:
    import cv2
except ImportError as exc:
    raise ImportError('Install opencv-python: pip install opencv-python') from exc

from models import MMFormerAnatomy
from preprocess.processing import validate_output_sizes
from utils.inference import predict_cine
from utils.checkpoint import ARCHITECTURE
from utils import MultiSourceNiftiDataset3D, load_cases_from_yaml, multiclass_dice_iou_hd95

VIEWS = ('2ch', '4ch', 'sa')
PALETTE = np.asarray([(0,0,0),(255,0,0),(0,255,0),(0,128,255),
                      (255,255,0),(255,0,255),(0,255,255),(255,128,0),
                      (128,0,255),(0,200,120)], dtype=np.uint8)


def safe_name(s):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s))[:140]


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def metric_scalar(x):
    x = to_numpy(x)
    return float(x.mean()) if x.size else float('nan')


def gray8(img):
    img = np.asarray(img, dtype=np.float32)
    lo, hi = float(img.min()), float(img.max())
    return np.zeros_like(img, dtype=np.uint8) if hi-lo < 1e-8 else (255*(img-lo)/(hi-lo)).clip(0,255).astype(np.uint8)


def label_rgb(y):
    return PALETTE[np.clip(y.astype(np.int64), 0, len(PALETTE)-1)]


def overlay(g, seg, a=.45):
    base = np.repeat(g[...,None], 3, axis=-1).astype(np.float32)
    m = (seg > 0)[...,None]
    return np.where(m, base*(1-a)+label_rgb(seg).astype(np.float32)*a, base).clip(0,255).astype(np.uint8)


def visualize(img, gt, pred, target):
    counts = (gt>0).reshape(gt.shape[0], -1).sum(1)
    idx = int(np.argmax(counts)) if np.any(counts) else gt.shape[0]//2
    im, y, p = gray8(img[idx]), gt[idx], pred[idx]
    raw = np.repeat(im[...,None], 3, axis=-1)
    spacer = np.full((raw.shape[0], 10, 3), 255, dtype=np.uint8)
    canvas = np.concatenate([raw,spacer,label_rgb(y),spacer,label_rgb(p),spacer,
                             overlay(im,y),spacer,overlay(im,p)], axis=1)
    target.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(target), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def normalize_item_image(images, source_order):
    """Dataset cines/volumes [1,T/D,H,W]; add a batch dimension per view."""
    result = {}
    for view in source_order:
        image = torch.as_tensor(images[view])
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError(f'{view}: expected [1,T/D,H,W], got {tuple(image.shape)}')
        result[view] = image.unsqueeze(0).float()
    return result


def evaluate_prediction(pred, target, view, classes):
    if view == 'sa':
        return multiclass_dice_iou_hd95(pred, target, classes)[:3]
    # Distances are spatial and must not use time as a third spatial axis.
    values = [multiclass_dice_iou_hd95(p, y, classes)[:3] for p, y in zip(pred, target)]
    return tuple(np.asarray(values).mean(axis=0))


def extract_mask(item, view, i, source_order):
    masks = item.get('masks')
    if masks is None:
        return None
    if isinstance(masks, dict):
        return masks.get(view)
    if isinstance(masks, (list,tuple)):
        return masks[i]
    if isinstance(masks, torch.Tensor):
        return masks[i] if masks.ndim == 4 else None
    raise TypeError('Unexpected item["masks"] type')


def make_model(config, order, classes, device):
    return MMFormerAnatomy(in_channels=1, source_order=order,
                           num_classes_by_source=classes,
                           base=config['base'], dim=config['dim'],
                           heads=config['heads'], token_grid=tuple(config['token_grid']),
                           intra_layers=config['intra_layers'],
                           inter_layers=config['inter_layers'],
                           temporal_size=config['temporal_size']).to(device)


def main():
    pa = argparse.ArgumentParser(description='YAML test/val evaluation for multi-view MMFormerAnatomy')
    pa.add_argument('--split-yaml', default='preprocess/matched.yaml')
    pa.add_argument('--split', choices=['train','val','test'], default='val')
    pa.add_argument('--checkpoint', required=True)
    pa.add_argument('--output-dir', default='results_mmformer')
    pa.add_argument('--lax-size', type=int, nargs=3, default=None, help='2CH/4CH K H W')
    pa.add_argument('--sax-size', type=int, nargs=3, default=None, help='SAX D H W')
    pa.add_argument('--source-order', nargs='+', default=None)
    pa.add_argument('--num-classes-json', default=None)
    pa.add_argument('--base', type=int, default=None)
    pa.add_argument('--dim', type=int, default=None)
    pa.add_argument('--heads', type=int, default=None)
    pa.add_argument('--token-grid', type=int, nargs=3, default=None)
    pa.add_argument('--intra-layers', type=int, default=None)
    pa.add_argument('--inter-layers', type=int, default=None)
    pa.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    pa.add_argument('--vis-per-source', type=int, default=3)
    pa.add_argument('--no-save-nifti', action='store_true')
    pa.add_argument('--seed', type=int, default=42, help='Fixed random seed for sampled modality masks')
    pa.add_argument('--eval-mode', choices=['both','full','random','all-seven'], default='both',
                    help='both: full and per-case random nonempty mask; all-seven: test all seven combinations')
    pa.add_argument('--random-repeats', type=int, default=1,
                    help='Independent random-mask draws per case (random/both modes)')
    pa.add_argument("--fp16", action="store_true", help="Enable CUDA FP16 mixed precision inference")
    args = pa.parse_args()
    if args.random_repeats < 1:
        pa.error('--random-repeats must be >= 1')
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.fp16 and device.type != "cuda":
        pa.error("--fp16 requires a CUDA device")
    print("Inference precision:", "FP16 AMP" if args.fp16 else "FP32")

    # Checkpoints written by the user's training script store model and metadata.
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    state = {k.removeprefix('module.'): v for k,v in state.items()}
    cases, info = load_cases_from_yaml(args.split_yaml, split=args.split, complete_cine=True)
    order = list(args.source_order or checkpoint.get('source_order') or info['source_order'])
    if order != list(VIEWS):
        raise ValueError(f'MMFormerAnatomy expects source_order {VIEWS}, found {order}')
    classes = (json.loads(args.num_classes_json) if args.num_classes_json else
               checkpoint.get('num_classes_by_source') or info['num_classes_by_source'])
    classes = {k:int(v) for k,v in classes.items()}
    sizes = validate_output_sizes(checkpoint.get('output_sizes') or info.get('output_sizes'))
    if sizes != validate_output_sizes(info.get('output_sizes')):
        raise ValueError('Checkpoint and preprocessing sizes differ; use matching window caches')
    if args.lax_size:
        sizes.update({'2ch': args.lax_size, '4ch': args.lax_size})
    if args.sax_size:
        sizes['sa'] = args.sax_size
    sizes = validate_output_sizes(sizes)
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Checkpoint is not a two-stage FiLM spatial-adapter model (v4). Retrain with the current architecture.')
    cfg = dict(checkpoint['model_config'])
    if any(sizes[v][0] != cfg['temporal_size'] for v in VIEWS[:2]):
        raise ValueError('Window size differs between checkpoint and requested preprocessing')
    for name in cfg:
        override = getattr(args, name, None)
        if override is not None:
            cfg[name] = override
    print('Split:', args.split, 'Cases:', len(cases), 'Classes:', classes, 'Per-view sizes:', sizes)
    print('Model hyperparameters:', cfg)
    if not cases:
        raise ValueError(f'No cases in YAML split {args.split!r}')
    dataset = MultiSourceNiftiDataset3D(cases=cases, output_sizes=sizes,
                                         source_num_classes=classes, augment=False,
                                         load_anno=args.split != 'test', complete_cine=True)
    model = make_model(cfg, order, classes, device)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError('Checkpoint does not match MMFormerAnatomy architecture. '
                           'Check that training used fusion/anatomy-token model, '
                           'label count and architecture arguments.\n' + str(exc)) from exc
    model.eval()
    all_masks = [(1,0,0), (0,1,0), (0,0,1), (1,1,0),
                 (1,0,1), (0,1,1), (1,1,1)]
    rng = random.Random(args.seed)
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    per_case = []
    vis_count = defaultdict(int)
    print('Evaluation mode:', args.eval_mode, '| random seed:', args.seed)
    print('NOTE: Missing-view predictions are not supported by this architecture; '
          'only present-view segmentations are evaluated in each scenario.')
    print('Model-space exports: LAX [H,W,1,T], SAX [H,W,D]; LAX metrics are framewise.')
    with torch.inference_mode():
        for idx in range(len(dataset)):
            item = dataset[idx]
            images = {v: image.to(device) for v, image in
                      normalize_item_image(item['image'], order).items()}
            identifier = safe_name(item.get('case_id', item.get('id', f'case_{idx:04d}')))
            scenarios = []
            if args.eval_mode in ('full', 'both'):
                scenarios.append(('full', (1,1,1)))
            if args.eval_mode in ('random', 'both'):
                for rep in range(args.random_repeats):
                    scenarios.append((f'random_{rep + 1}', rng.choice(all_masks[:-1])))
            if args.eval_mode == 'all-seven':
                scenarios.extend((f'mask_{"".join(map(str, m))}', m) for m in all_masks)
            for scenario, mask_values in scenarios:
                present = torch.tensor(mask_values, dtype=torch.bool, device=device).unsqueeze(0)
                predictions = predict_cine(model, images, present, fp16=args.fp16)
                for vi, view in enumerate(order):
                    if not mask_values[vi]:
                        continue  # Never score absent-view zero placeholders.
                    pred = predictions[view][0]
                    gt = extract_mask(item, view, vi, order)
                    scores = {'dice': None, 'iou': None, 'hd95': None}
                    if gt is not None:
                        gt = torch.as_tensor(gt).long().cpu()
                        if tuple(gt.shape) != tuple(pred.shape):
                            raise ValueError(f'{view}: prediction and target shapes disagree')
                        values = evaluate_prediction(pred, gt, view, classes[view])
                        for name, value in zip(('dice','iou','hd95'), values[:3]):
                            score = metric_scalar(value)
                            if np.isfinite(score):
                                stats[scenario][view][name].append(score)
                            scores[name] = score if np.isfinite(score) else None
                    stem = f'{idx:04d}_{identifier}_{view}'
                    per_case.append({'case_index':idx, 'case_id':identifier,
                                     'scenario':scenario, 'mask':''.join(map(str,mask_values)),
                                     'view':view, **scores})
                    if not args.no_save_nifti:
                        folder = outdir / scenario
                        folder.mkdir(parents=True, exist_ok=True)
                        array = np.moveaxis(pred.numpy().astype(np.uint8), 0, -1)
                        if view != 'sa':
                            array = array[:, :, None, :]  # NIfTI time is dimension four.
                        nib.save(nib.Nifti1Image(array, np.eye(4)),
                                 str(folder / f'{stem}_model_space.nii.gz'))
                    vis_key = (scenario, view)
                    if gt is not None and vis_count[vis_key] < args.vis_per_source:
                        visualize(images[view][0,0].cpu().numpy(), gt.numpy(), pred.numpy(),
                                  outdir / 'visualization' / scenario / view / f'{stem}.png')
                        vis_count[vis_key] += 1
            if (idx+1) % 10 == 0 or idx+1 == len(dataset):
                print(f'Processed {idx+1}/{len(dataset)}')

    report = {}
    for scenario in sorted(stats):
        report[scenario] = {}
        print(f'\n--- {scenario} ---')
        for view in order:
            result = {name: (float(np.mean(stats[scenario][view][name]))
                              if stats[scenario][view][name] else None)
                      for name in ('dice','iou','hd95')}
            result['cases_evaluated'] = len(stats[scenario][view]['dice'])
            report[scenario][view] = result
            print(view, result)
    if per_case:
        with (outdir/'per_case_metrics.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(per_case[0]))
            writer.writeheader()
            writer.writerows(per_case)
    (outdir/'metrics.json').write_text(json.dumps({
        'split': args.split, 'evaluation_mode': args.eval_mode,
        'precision': 'fp16' if args.fp16 else 'fp32',
        'random_seed': args.seed, 'random_repeats': args.random_repeats,
        'output_sizes':{v:list(s) for v,s in sizes.items()},
        'lax_metrics':'mean over 2D frames', 'network_input':'centered LAX windows + shared 3D SAX',
        'sax_aggregation':'mean window probabilities', 'pairing':'nearest normalized sequence position',
        'results':report
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Saved outputs to:', outdir)


if __name__ == '__main__':
    main()
