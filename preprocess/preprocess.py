"""Match cases and cache per-view arrays with explicit time/spatial axes."""
import argparse
import re
import sys
from pathlib import Path

# Support both `python preprocess/preprocess.py` and `python -m preprocess.preprocess`.
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from preprocess.processing import prepare_nifti, validate_output_sizes

SOURCE_FOLDERS = {'2ch': '2CH', '4ch': '4CH', 'sa': 'SAX'}
SPLITS = {'train': '_TR', 'val': '_VAL', 'test': '_TST'}


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
    parser.add_argument('--output-yaml', default='preprocess/matched.yaml')
    parser.add_argument('--output-dir', default='preprocess/processed')
    parser.add_argument('--lax-size', nargs=3, type=int, default=(64, 128, 128), metavar=('T', 'H', 'W'))
    parser.add_argument('--sax-size', nargs=3, type=int, default=(128, 128, 128), metavar=('D', 'H', 'W'))
    parser.add_argument('--manifest-only', action='store_true',
                        help='Keep raw NIfTI paths; the Dataset applies the same preprocessing online')
    args = parser.parse_args()
    sizes = validate_output_sizes({'2ch': args.lax_size, '4ch': args.lax_size, 'sa': args.sax_size})
    root, output_dir = Path(args.data_root).resolve(), Path(args.output_dir).resolve()
    splits = {}
    for split, suffix in SPLITS.items():
        files = {v: {k: collect_files(root, v, suffix, k) for k in ('image', 'anno')}
                 for v in SOURCE_FOLDERS}
        ids = set.intersection(*(set(files[v]['image']) for v in SOURCE_FOLDERS))
        if split != 'test':
            ids &= set.intersection(*(set(files[v]['anno']) for v in SOURCE_FOLDERS))
        records = []
        for case_id in sorted(ids, key=int):
            record = {'case_id': case_id}
            for view in SOURCE_FOLDERS:
                image_path = files[view]['image'][case_id]
                label_path = files[view]['anno'].get(case_id) if split != 'test' else None
                if args.manifest_only:
                    record[view] = {'image': image_path}
                    if label_path:
                        record[view]['anno'] = label_path
                    continue
                image, affine, shape = prepare_nifti(image_path, view, sizes[view])
                arrays = dict(image=image, affine=affine, original_shape=np.array(shape),
                              view=np.array(view), layout=np.array('DHW' if view == 'sa' else 'THW'))
                if label_path:
                    label, _, label_shape = prepare_nifti(label_path, view, sizes[view], is_label=True)
                    if tuple(label_shape) != tuple(shape):
                        raise ValueError(f'Image/label shape mismatch for {case_id}/{view}')
                    classes = {'2ch': 3, '4ch': 6, 'sa': 4}[view]
                    if label.min() < 0 or label.max() >= classes:
                        raise ValueError(f'Invalid labels for {case_id}/{view}')
                    arrays['label'] = label.astype(np.uint8)
                destination = output_dir / split / f'{view}_{case_id}.npz'
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(destination, **arrays)
                record[view] = {'image': str(destination)}
                if label_path:
                    record[view]['anno'] = str(destination)
            records.append(record)
            print(f'{split}: {case_id}', flush=True)
        splits[split] = records
    output = {'dataset_info': {
        'data_root': str(root), 'source_order': list(SOURCE_FOLDERS),
        'num_classes_by_source': {'2ch': 3, '4ch': 6, 'sa': 4},
        'output_sizes': {v: list(s) for v, s in sizes.items()},
        'array_layouts': {'2ch': 'THW', '4ch': 'THW', 'sa': 'DHW'},
        'sample_counts': {split: len(records) for split, records in splits.items()},
    }, 'splits': splits}
    destination = Path(args.output_yaml)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(output, sort_keys=False, allow_unicode=True), encoding='utf-8')
    print(f'Saved {destination}; sizes={sizes}')


if __name__ == '__main__':
    main()
