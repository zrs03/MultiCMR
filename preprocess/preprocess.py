import argparse
import json
import os
from glob import glob
from typing import Dict, List, Sequence, Tuple
import yaml
import numpy as np

NII_EXTENSIONS = (".nii", ".nii.gz")


def _strip_nii_suffix(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def collect_dataset_records(
    data_root: str,
    source_names: Sequence[str],
    image_dirname: str = "image",
    label_dirname: str = "seg",
) -> Tuple[List[dict], List[str]]:
    """Scan dataset and return paired records with unique case IDs."""
    records = []
    case_ids_set = set()

    for source_id, source in enumerate(source_names):
        source_root = os.path.join(data_root, source)
        image_root = os.path.join(source_root, image_dirname)
        label_root = os.path.join(source_root, label_dirname)

        if not os.path.isdir(image_root) or not os.path.isdir(label_root):
            print(f"Warning: Directory not found for source {source}: {image_root} or {label_root}")
            continue

        image_paths = []
        label_paths = []
        for extension in NII_EXTENSIONS:
            image_paths.extend(glob(os.path.join(image_root, "**", f"*{extension}"), recursive=True))
            label_paths.extend(glob(os.path.join(label_root, "**", f"*{extension}"), recursive=True))

        image_map = {_strip_nii_suffix(os.path.basename(path)): path for path in sorted(image_paths)}
        label_map = {_strip_nii_suffix(os.path.basename(path)): path for path in sorted(label_paths)}

        shared_names = sorted(set(image_map) & set(label_map))
        for name in shared_names:
            img_p = os.path.relpath(image_map[name], data_root)
            lbl_p = os.path.relpath(label_map[name], data_root)
            records.append({
                "case_id": name,
                "source": source,
                "source_id": source_id,
                "image_path": img_p,
                "label_path": lbl_p,
            })
            case_ids_set.add(name)

    return records, sorted(list(case_ids_set))


def split_case_ids(
    case_ids: List[str],
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Patient-level split (Train : Val : Test) to prevent data leakage."""
    r_train, r_val, r_test = split_ratio
    total_ratio = r_train + r_val + r_test
    r_train, r_val, r_test = r_train / total_ratio, r_val / total_ratio, r_test / total_ratio

    rng = np.random.default_rng(seed)
    shuffled_ids = list(case_ids)
    rng.shuffle(shuffled_ids)

    n_total = len(shuffled_ids)
    if n_total == 0:
        return {"train": [], "val": [], "test": []}

    n_train = int(round(n_total * r_train))
    n_val = int(round(n_total * r_val))
    # Ensure train + val + test == n_total
    n_test = n_total - n_train - n_val

    # Edge cases handling
    if n_total >= 3:
        if n_val == 0:
            n_val = 1
            n_train = max(1, n_train - 1)
        if n_test == 0:
            n_test = 1
            n_train = max(1, n_train - 1)

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def load_id_mapping(mapping_path: str) -> Dict[str, int]:
    if not mapping_path or not os.path.exists(mapping_path):
        return {}
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): int(v) for k, v in data.items()}


def create_dataset_split_yaml(
    data_root: str,
    source_order: Sequence[str] = ("2ch", "4ch", "sa"),
    image_dirname: str = "image",
    label_dirname: str = "seg",
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    num_classes_by_source: Dict[str, int] = None,
    id_mapping_path: str = "",
    output_yaml: str = "preprocess/dataset_split.yaml",
    seed: int = 42,
) -> dict:
    if num_classes_by_source is None:
        num_classes_by_source = {"2ch": 3, "4ch": 5, "sa": 4}

    records, unique_case_ids = collect_dataset_records(
        data_root=data_root,
        source_names=source_order,
        image_dirname=image_dirname,
        label_dirname=label_dirname,
    )

    id_splits = split_case_ids(unique_case_ids, split_ratio=split_ratio, seed=seed)
    train_set = set(id_splits["train"])
    val_set = set(id_splits["val"])
    test_set = set(id_splits["test"])

    splits_data = {"train": [], "val": [], "test": []}
    for rec in records:
        cid = rec["case_id"]
        if cid in train_set:
            splits_data["train"].append(rec)
        elif cid in val_set:
            splits_data["val"].append(rec)
        elif cid in test_set:
            splits_data["test"].append(rec)

    # Load existing id_mapping
    id_mapping = load_id_mapping(id_mapping_path)

    yaml_data = {
        "dataset_info": {
            "data_root": os.path.abspath(data_root),
            "source_order": list(source_order),
            "num_classes_by_source": num_classes_by_source,
            "split_ratio": {
                "train": float(split_ratio[0]),
                "val": float(split_ratio[1]),
                "test": float(split_ratio[2]),
            },
            "case_counts": {
                "total": len(unique_case_ids),
                "train": len(id_splits["train"]),
                "val": len(id_splits["val"]),
                "test": len(id_splits["test"]),
            },
            "sample_counts": {
                "total": len(records),
                "train": len(splits_data["train"]),
                "val": len(splits_data["val"]),
                "test": len(splits_data["test"]),
            },
        },
        "id_mapping": id_mapping,
        "split_case_ids": id_splits,
        "splits": splits_data,
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_yaml)), exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True, indent=2)

    print(f"[Preprocess] Successfully generated dataset split YAML at: {output_yaml}")
    print(f"  - Total unique cases: {len(unique_case_ids)} (Train: {len(id_splits['train'])}, Val: {len(id_splits['val'])}, Test: {len(id_splits['test'])})")
    print(f"  - Total sample volumes: {len(records)} (Train: {len(splits_data['train'])}, Val: {len(splits_data['val'])}, Test: {len(splits_data['test'])})")
    print(f"  - Integrated id_mapping entries: {len(id_mapping)}")
    return yaml_data


def main():
    parser = argparse.ArgumentParser(description="MultiCMR Preprocess: Split dataset (8:1:1) and save YAML with id_mapping")
    parser.add_argument("--data-root", type=str, required=True, help="Root folder containing sequence subdirectories")
    parser.add_argument("--source-order", nargs="+", default=["2ch", "4ch", "sa"], help="List of sequence sources")
    parser.add_argument("--image-dirname", type=str, default="image", help="Subdirectory name for images")
    parser.add_argument("--label-dirname", type=str, default="seg", help="Subdirectory name for labels")
    parser.add_argument("--split-ratio", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train:Val:Test ratio (e.g. 0.8 0.1 0.1)")
    parser.add_argument("--num-classes-json", type=str, default='{"2ch":3,"4ch":5,"sa":4}', help="Classes count per sequence")
    parser.add_argument("--id-mapping", type=str, default="rawdata/id_mapping.json", help="Path to input id_mapping.json to merge")
    parser.add_argument("--output-yaml", type=str, default="preprocess/dataset_split.yaml", help="Path to output YAML file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split")
    args = parser.parse_args()

    num_classes = json.loads(args.num_classes_json) if args.num_classes_json else {"2ch": 3, "4ch": 5, "sa": 4}

    create_dataset_split_yaml(
        data_root=args.data_root,
        source_order=args.source_order,
        image_dirname=args.image_dirname,
        label_dirname=args.label_dirname,
        split_ratio=tuple(args.split_ratio),
        num_classes_by_source=num_classes,
        id_mapping_path=args.id_mapping,
        output_yaml=args.output_yaml,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
