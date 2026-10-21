"""Match cases and cache per-view arrays with explicit time/spatial axes."""
import argparse
import json
import re
import sys
from pathlib import Path

# Support both `python preprocess/preprocess.py` and `python -m preprocess.preprocess`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from preprocess.processing import prepare_nifti, validate_output_sizes, window_indices, paired_centers, validate_sax_geometry, sax_context_indices

SOURCE_FOLDERS = {'2ch': '2CH', '4ch': '4CH', 'sa': 'SAX'}
SPLITS = {'train': '_TR', 'val': '_VAL', 'test': '_TST'}


PHASE_TABLES = {'train': 'sax_slice_info.json', 'val': 'id_slice_info_valid.json',
                'test': 'sax_slice_info_test.json'}


def resolve_phase_table(args, split):
    """Split override > explicit common override > official file in data root."""
    override = getattr(args, f'sax_phase_info_{split}', None) or args.sax_phase_info
    path = Path(override) if override else Path(args.data_root) / PHASE_TABLES[split]
    if not path.is_file():
        raise FileNotFoundError(f'{split}: SAX phase table missing: {path.resolve()}; '
                                f'use --sax-phase-info-{split} to override')
    return path.resolve()


def collect_files(root, source, suffix, kind):
    folder = root / (SOURCE_FOLDERS[source] + suffix) / kind
    result = {}
    for path in sorted(folder.glob('*.nii*')):
        match = re.fullmatch(r'CINE_(?:2CH|4CH|SAX)_(\d+)\.nii(?:\.gz)?', path.name)
        if match:
            case_id = match.group(1)
            if case_id in result:
                raise ValueError(f'Duplicate case {case_id} in {folder}')
            result[case_id] = str(path.resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='rawdata/CINE_MULTI')
    parser.add_argument('--output-yaml', default='preprocess/matched_sax_context.yaml')
    parser.add_argument('--output-dir', default='preprocess/processed_sax_context')
    parser.add_argument('--lax-size', nargs=3, type=int, default=(5, 128, 128), metavar=('K', 'H', 'W'))
    parser.add_argument('--sax-size', nargs=3, type=int, default=(5, 128, 128), metavar=('K', 'H', 'W'))
    parser.add_argument('--sax-phase-info', default=None, help='Explicit common phase-table override; default uses per-split JSON under data-root')
    for split in SPLITS:
        parser.add_argument(f'--sax-phase-info-{split}', default=None, help='Optional split-specific phase table')
    args = parser.parse_args()
    sizes = validate_output_sizes({'2ch': args.lax_size, '4ch': args.lax_size, 'sa': args.sax_size})
    root, output_dir = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    splits, case_splits = {}, {}
    for split, suffix in SPLITS.items():
        files = {v: {k: collect_files(root, v, suffix, k) for k in ('image', 'anno')}
                 for v in SOURCE_FOLDERS}
        ids = set.intersection(*(set(files[v]['image']) for v in SOURCE_FOLDERS))
        if split != 'test':
            ids &= set.intersection(*(set(files[v]['anno']) for v in SOURCE_FOLDERS))
        records, case_records = [], []
        phase_info = {}
        if ids:
            path = resolve_phase_table(args, split)
            print(f'[{split}] SAX phase table: {path}', flush=True)
            with open(path) as f:
                phase_info = {str(int(k)): v for k, v in json.load(f).items()}
        for case_id in sorted(ids, key=int):
            record = {'case_id': case_id}
            cached = {}
            for view in SOURCE_FOLDERS:
                image_path = files[view]['image'][case_id]
                label_path = files[view]['anno'].get(case_id) if split != 'test' else None
                image, affine, shape = prepare_nifti(image_path, view, sizes[view])
                arrays = dict(image=image, affine=affine, original_shape=np.array(shape),
                              view=np.array(view), layout=np.array('FHW' if view == 'sa' else 'THW'))
                if label_path:
                    label, _, label_shape = prepare_nifti(label_path, view, sizes[view], is_label=True)
                    if tuple(label_shape) != tuple(shape):
                        raise ValueError(f'Image/label shape mismatch for {case_id}/{view}')
                    classes = {'2ch': 3, '4ch': 6, 'sa': 4}[view]
                    if label.min() < 0 or label.max() >= classes:
                        raise ValueError(f'Invalid labels for {case_id}/{view}')
                    arrays['label'] = label.astype(np.uint8)
                if view == 'sa':
                    if str(int(case_id)) not in phase_info:
                        raise ValueError(f'Missing SAX phase count for {split}/{case_id}; supply --sax-phase-info-{split}')
                    phases = phase_info[str(int(case_id))]
                    try:
                        slices = validate_sax_geometry(image.shape[0], phases, allow_partial=True)
                    except ValueError as exc:
                        raise ValueError(
                            f'{split}/{case_id}: {image_path}; original_shape={shape}; '
                            f'phase_table={path}; phases={phases}. {exc}. '
                            'Check acquisition metadata; do not guess a divisor. '
                            'Run python -m preprocess.audit_geometry to audit all cases.'
                        ) from exc
                    valid_frames = slices * phases
                    tail_frames = image.shape[0] - valid_frames
                    if tail_frames:
                        foreground = (f", tail foreground pixels={np.count_nonzero(arrays['label'][valid_frames:])}"
                                      if 'label' in arrays else '')
                        print(f'WARNING {split}/{case_id}: SAX keeps {valid_frames}/{image.shape[0]} frames '
                              f'(Z={slices}, T={phases}); excluding {tail_frames} trailing frames from '
                              f'training contexts; inference tail will be background{foreground}', flush=True)
                    # Preserve original images/labels for export and honest evaluation.
                    arrays.update(sax_valid_frames=np.array(valid_frames), sax_tail_frames=np.array(tail_frames))
                    arrays.update(sax_phases=np.array(phases), sax_slices=np.array(slices),
                                  flat_order=np.array('slice_phase'))
                cached[view] = arrays
                destination = output_dir / split / case_id / f'{view}_full.npz'
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **arrays)
                record[view] = {'image': str(destination)}
                if label_path:
                    record[view]['anno'] = str(destination)
            case_records.append(record)
            lengths = {v: cached[v]['image'].shape[0] for v in ('2ch', '4ch')}
            lengths['sa'] = phases
            centers = paired_centers(lengths)
            for index, center in enumerate(centers):
                # LAX windows are stored once per paired center and referenced by all slices.
                lax_record = {}
                for view in ('2ch', '4ch'):
                    full = cached[view]
                    indices = window_indices(center[view], lengths[view], sizes[view][0])
                    arrays = dict(image=full['image'][indices], affine=full['affine'],
                                  original_shape=full['original_shape'], view=np.array(view),
                                  layout=np.array('KHW'), center_index=np.array(center[view]),
                                  frame_indices=indices)
                    if 'label' in full:
                        arrays['label'] = full['label'][center[view]]
                    destination = output_dir / split / case_id / f'{view}_w{index:04d}.npz'
                    np.savez_compressed(destination, **arrays)
                    lax_record[view] = {'image': str(destination)}
                    if 'label' in arrays:
                        lax_record[view]['anno'] = str(destination)
                for z in range(slices):
                    t = center['sa']
                    indices = sax_context_indices(t, z, phases, slices)
                    destination = output_dir / split / case_id / f'sa_t{t:03d}_z{z:03d}.npz'
                    full = cached['sa']
                    if not destination.exists() or index == 0 or centers[index-1]['sa'] != t:
                        arrays = dict(image=full['image'][indices], affine=full['affine'],
                                      original_shape=full['original_shape'], view=np.array('sa'),
                                      layout=np.array('KHW'), frame_indices=indices,
                                      center_phase=np.array(t), center_slice=np.array(z),
                                      sax_phases=np.array(phases), sax_slices=np.array(slices),
                                      flat_order=np.array('slice_phase'))
                        if 'label' in full:
                            arrays['label'] = full['label'][indices[0]]
                        np.savez_compressed(destination, **arrays)
                    sax_record = {'image': str(destination)}
                    if 'label' in full:
                        sax_record['anno'] = str(destination)
                    records.append(dict(case_id=case_id, sample_id=f'{case_id}_w{index:04d}_z{z:03d}',
                                        center_indices=dict(center, sa_slice=z), sa=sax_record, **lax_record))
            print(f'{split}: {case_id}, {len(centers) * slices} paired samples; SAX Z={slices}, T={phases}', flush=True)
        splits[split] = records
        case_splits[split] = case_records
    output = {'dataset_info': {
        'schema': 'sax_context_v2', 'window_size': sizes['2ch'][0],
        'pairing': 'nearest_normalized_position', 'padding': {'lax': 'edge', 'sa': 'reflect'}, 'sax_flat_order': 'slice_phase',
        'data_root': str(root), 'source_order': list(SOURCE_FOLDERS),
        'num_classes_by_source': {'2ch': 3, '4ch': 6, 'sa': 4},
        'output_sizes': {v: list(s) for v, s in sizes.items()},
        'array_layouts': {'2ch': 'KHW', '4ch': 'KHW', 'sa': 'KHW'},
        'sample_counts': {split: len(records) for split, records in splits.items()},
    }, 'splits': splits, 'case_splits': case_splits}
    destination = Path(args.output_yaml)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'Saved {destination}; sizes={sizes}')


if __name__ == '__main__':
    main()
