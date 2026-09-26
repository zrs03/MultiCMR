import argparse
import json
import os
from glob import glob
from typing import Dict, List, Sequence, Tuple
import yaml

NII_EXTENSIONS = (".nii", ".nii.gz")

# Exact folder mapping:
# 2ch -> 2CH_TR, 2CH_VAL, 2CH_TST
# 4ch -> 4CH_TR, 4CH_VAL, 4CH_TST
# sa  -> SAX_TR, SAX_VAL, SAX_TST
SOURCE_PREFIX_MAP = {
    "2ch": "2CH",
    "4ch": "4CH",
    "sa": "SAX",
    "sax": "SAX",
    "2CH": "2CH",
    "4CH": "4CH",
    "SAX": "SAX",
}

SPLIT_SUFFIX_MAP = {
    "train": "_TR",
    "val": "_VAL",
    "test": "_TST",
}


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def find_subfolder(parent_dir: str, candidates: Sequence[str]) -> str:
    for name in candidates:
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p):
            return name
    return ""


def scan_prepartitioned_folders(
    data_root: str,
    source_order: Sequence[str] = ("2ch", "4ch", "sa"),
    image_dirnames: Sequence[str] = ("image", "images"),
    label_dirnames: Sequence[str] = ("anno", "seg", "label", "labels"),
) -> Tuple[Dict[str, List[dict]], Dict[str, int]]:
    """
    Scans predefined directories:
      - Train: 2CH_TR, 4CH_TR, SAX_TR (requires image/ and anno/)
      - Val:   2CH_VAL, 4CH_VAL, SAX_VAL (requires image/ and anno/)
      - Test:  2CH_TST, 4CH_TST, SAX_TST (only requires image/, no anno/)
    """
    splits = {"train": [], "val": [], "test": []}

    for source_id, source in enumerate(source_order):
        prefix = SOURCE_PREFIX_MAP.get(source, source.upper())

        for split_key, split_suffix in SPLIT_SUFFIX_MAP.items():
            folder_name = f"{prefix}{split_suffix}"
            folder_abs = os.path.join(data_root, folder_name)

            if not os.path.isdir(folder_abs):
                print(f"[Warning] Folder not found: {folder_abs}")
                continue

            img_dir = find_subfolder(folder_abs, image_dirnames)
            lbl_dir = find_subfolder(folder_abs, label_dirnames)

            if not img_dir:
                print(f"[Warning] Missing image subfolder in {folder_abs}")
                continue

            if split_key in ("train", "val") and not lbl_dir:
                print(f"[Warning] Missing annotation subfolder (anno/) in {folder_abs}")
                continue

            img_abs = os.path.join(folder_abs, img_dir)
            img_files = []
            for ext in NII_EXTENSIONS:
                img_files.extend(glob(os.path.join(img_abs, f"*{ext}")))
            img_map = {_strip_nii_suffix(os.path.basename(p)): p for p in sorted(img_files)}

            if lbl_dir:
                lbl_abs = os.path.join(folder_abs, lbl_dir)
                lbl_files = []
                for ext in NII_EXTENSIONS:
                    lbl_files.extend(glob(os.path.join(lbl_abs, f"*{ext}")))
                lbl_map = {_strip_nii_suffix(os.path.basename(p)): p for p in sorted(lbl_files)}

                paired_cases = sorted(set(img_map.keys()) & set(lbl_map.keys()))
                for case_id in paired_cases:
                    rel_img = os.path.relpath(img_map[case_id], data_root)
                    rel_lbl = os.path.relpath(lbl_map[case_id], data_root)
                    splits[split_key].append({
                        "case_id": case_id,
                        "source": source,
                        "source_id": source_id,
                        "folder": folder_name,
                        "image_path": rel_img,
                        "label_path": rel_lbl,
                    })
            else:
                # Test set without annotations (_TST): image only
                for case_id, img_p in img_map.items():
                    rel_img = os.path.relpath(img_p, data_root)
                    splits[split_key].append({
                        "case_id": case_id,
                        "source": source,
                        "source_id": source_id,
                        "folder": folder_name,
                        "image_path": rel_img,
                        "label_path": "",
                    })

    counts = {k: len(v) for k, v in splits.items()}
    return splits, counts


def generate_yaml_from_folders(
    data_root: str,
    source_order: Sequence[str] = ("2ch", "4ch", "sa"),
    num_classes_by_source: Dict[str, int] = None,
    output_yaml: str = "preprocess/dataset_split.yaml",
) -> dict:
    if num_classes_by_source is None:
        num_classes_by_source = {"2ch": 3, "4ch": 5, "sa": 4}

    splits, counts = scan_prepartitioned_folders(
        data_root=data_root,
        source_order=source_order,
    )

    yaml_data = {
        "dataset_info": {
            "data_root": os.path.abspath(data_root) if data_root else "",
            "source_order": list(source_order),
            "num_classes_by_source": num_classes_by_source,
            "sample_counts": {
                "total": sum(counts.values()),
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
            },
        },
        "splits": splits,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_yaml)), exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True, indent=2)

    print(f"\n[Preprocess] Successfully scanned folders and generated YAML at: {output_yaml}")
    print(f"  - Data Root: {os.path.abspath(data_root)}")
    print(f"  - Sources: {list(source_order)}")
    print(f"  - Folders Scanned: 2CH_TR/VAL/TST, 4CH_TR/VAL/TST, SAX_TR/VAL/TST")
    print(f"  - Total Samples: {sum(counts.values())} (Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']})")
    print(f"  - Test set mode: Image only (no anno/ required)\n")
    return yaml_data


def main():
    parser = argparse.ArgumentParser(
        description="MultiCMR: Scan 2CH_TR/VAL/TST, 4CH_TR/VAL/TST, SAX_TR/VAL/TST folders and generate YAML"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root folder containing 2CH_TR, 2CH_VAL, 2CH_TST, 4CH_*, SAX_* folders",
    )
    parser.add_argument(
        "--source-order",
        nargs="+",
        default=["2ch", "4ch", "sa"],
        help="Sequence source keys to include (default: 2ch 4ch sa)",
    )
    parser.add_argument(
        "--num-classes-json",
        type=str,
        default='{"2ch":3,"4ch":6,"sa":4}',
        help="JSON string specifying number of classes per source",
    )
    parser.add_argument(
        "--output-yaml",
        type=str,
        default="preprocess/dataset_split.yaml",
        help="Path where output YAML file will be saved (default: preprocess/dataset_split.yaml)",
    )
    args = parser.parse_args()

    num_classes = json.loads(args.num_classes_json) if args.num_classes_json else {"2ch": 3, "4ch": 5, "sa": 4}

    generate_yaml_from_folders(
        data_root=args.data_root,
        source_order=args.source_order,
        num_classes_by_source=num_classes,
        output_yaml=args.output_yaml,
    )


if __name__ == "__main__":
    main()
