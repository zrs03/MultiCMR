"""Read headers and phase tables without writing caches or guessing geometry."""
import argparse
import json
from pathlib import Path

import nibabel as nib

from .preprocess import SPLITS, SOURCE_FOLDERS, collect_files, resolve_phase_table
from .processing import validate_sax_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='rawdata/CINE_MULTI')
    parser.add_argument('--sax-phase-info', default=None)
    for split in SPLITS:
        parser.add_argument(f'--sax-phase-info-{split}')
    args = parser.parse_args()
    errors = 0
    for split, suffix in SPLITS.items():
        files = {v: collect_files(Path(args.data_root), v, suffix, 'image') for v in SOURCE_FOLDERS}
        if not files['sa']:
            continue
        table = resolve_phase_table(args, split)
        with open(table) as f:
            phases_by_id = {str(int(k)): v for k, v in json.load(f).items()}
        print(f'[{split}] phase table: {Path(table).resolve()}')
        for cid, path in sorted(files['sa'].items(), key=lambda pair: int(pair[0])):
            shapes = {v: nib.load(paths[cid]).shape for v, paths in files.items() if cid in paths}
            phases = phases_by_id.get(str(int(cid)))
            try:
                if len(shapes['sa']) != 3:
                    raise ValueError('Expected flattened 3D SAX array')
                slices = validate_sax_geometry(shapes['sa'][-1], phases)
                status = f'SAX divisible: Z={slices}, T={phases} (ordering still requires metadata)'
            except ValueError as exc:
                status = f'ERROR: {exc}'
                errors += 1
            print(f'{cid}: shapes={shapes}; table_T={phases}; {status}')
    print(f'Invalid SAX counts: {errors}. Divisibility alone does not prove correct phase count or frame order.')


if __name__ == '__main__':
    main()
