import os
import re
import yaml
from glob import glob


# ============================================================
# Configuration
# ============================================================

DATA_ROOT = "./rawdata/CINE_MULTI"
OUTPUT_YAML = "./preprocess/matched.yaml"

SOURCE_FOLDERS = {
    "2ch": "2CH",
    "4ch": "4CH",
    "sa": "SAX",
}

SPLITS = {
    "train": "_TR",
    "val": "_VAL",
    "test": "_TST",
}


# ============================================================
# Extract case ID
# ============================================================

def get_case_id(filename):
    """
    Examples:
        CINE_2CH_024.nii.gz -> 024
        CINE_4CH_024.nii.gz -> 024
        CINE_SAX_024.nii.gz -> 024
    """
    basename = os.path.basename(filename)

    match = re.search(
        r"CINE_(?:2CH|4CH|SAX)_(\d+)\.nii(?:\.gz)?$",
        basename,
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# Collect files for one source and split
# ============================================================

def collect_files(source, split_suffix, data_type):
    """
    data_type:
        "image" -> image/
        "anno"  -> anno/

    Returns:
        {
            "024": "path/to/CINE_2CH_024.nii.gz",
            ...
        }
    """

    folder = SOURCE_FOLDERS[source] + split_suffix

    data_dir = os.path.join(
        DATA_ROOT,
        folder,
        data_type,
    )

    if not os.path.isdir(data_dir):
        print(f"[Warning] Directory not found: {data_dir}")
        return {}

    files = []
    files.extend(glob(os.path.join(data_dir, "*.nii")))
    files.extend(glob(os.path.join(data_dir, "*.nii.gz")))

    result = {}

    for path in sorted(files):
        case_id = get_case_id(path)

        if case_id is None:
            print(f"[Warning] Cannot parse case ID: {path}")
            continue

        if case_id in result:
            print(
                f"[Warning] Duplicate case ID {case_id} "
                f"for source={source}, split={split_suffix}"
            )
            continue

        result[case_id] = os.path.abspath(path)

    return result


# ============================================================
# Main
# ============================================================

def main():

    all_splits = {}
    sample_counts = {}

    for split_name, split_suffix in SPLITS.items():

        print()
        print("=" * 70)
        print(f"Processing split: {split_name} ({split_suffix})")
        print("=" * 70)

        source_files = {}

        # ----------------------------------------------------
        # Collect images for all sources
        # ----------------------------------------------------

        for source in SOURCE_FOLDERS:

            source_files[source] = {
                "image": collect_files(
                    source,
                    split_suffix,
                    "image",
                ),
                "anno": {},
            }

            # Train/Val require annotations.
            # Test only requires images.
            if split_name != "test":
                source_files[source]["anno"] = collect_files(
                    source,
                    split_suffix,
                    "anno",
                )

            print(
                f"{source:>4}: "
                f"image={len(source_files[source]['image']):4d}, "
                f"anno={len(source_files[source]['anno']):4d}"
            )

        # ----------------------------------------------------
        # Find matched case IDs
        # ----------------------------------------------------

        # Every split requires images from all three sources.
        common_ids = (
            set(source_files["2ch"]["image"].keys())
            & set(source_files["4ch"]["image"].keys())
            & set(source_files["sa"]["image"].keys())
        )

        # Train/Val additionally require annotations
        # from all three sources.
        if split_name != "test":
            common_ids &= (
                set(source_files["2ch"]["anno"].keys())
                & set(source_files["4ch"]["anno"].keys())
                & set(source_files["sa"]["anno"].keys())
            )

        common_ids = sorted(
            common_ids,
            key=lambda x: int(x),
        )

        print(f"Matched cases: {len(common_ids)}")

        # ----------------------------------------------------
        # Build cases
        # ----------------------------------------------------

        matched_cases = []

        for case_id in common_ids:

            case = {
                "case_id": case_id,
                "2ch": {
                    "image": source_files["2ch"]["image"][case_id],
                },
                "4ch": {
                    "image": source_files["4ch"]["image"][case_id],
                },
                "sa": {
                    "image": source_files["sa"]["image"][case_id],
                },
            }

            # Train/Val: add annotation paths
            if split_name != "test":
                case["2ch"]["anno"] = source_files["2ch"]["anno"][case_id]
                case["4ch"]["anno"] = source_files["4ch"]["anno"][case_id]
                case["sa"]["anno"] = source_files["sa"]["anno"][case_id]

            matched_cases.append(case)

        all_splits[split_name] = matched_cases
        sample_counts[split_name] = len(matched_cases)

    # ========================================================
    # Generate YAML
    # ========================================================

    output = {
        "dataset_info": {
            "data_root": os.path.abspath(DATA_ROOT),
            "source_order": ["2ch", "4ch", "sa"],
            "sample_counts": {
                "train": sample_counts["train"],
                "val": sample_counts["val"],
                "test": sample_counts["test"],
                "total": sum(sample_counts.values()),
            },
        },
        "splits": all_splits,
    }

    output_dir = os.path.dirname(
        os.path.abspath(OUTPUT_YAML)
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(OUTPUT_YAML, "w", encoding="utf-8") as f:
        yaml.dump(
            output,
            f,
            sort_keys=False,
            allow_unicode=True,
            indent=2,
        )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Train: {sample_counts['train']}")
    print(f"Val:   {sample_counts['val']}")
    print(f"Test:  {sample_counts['test']}")
    print(f"Total: {sum(sample_counts.values())}")
    print(f"YAML:  {OUTPUT_YAML}")


if __name__ == "__main__":
    main()