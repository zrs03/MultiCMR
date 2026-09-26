import argparse
import json
import logging
import os
import nibabel as nib
import numpy as np
from scipy.stats import mode

LV_BLOOD_POOL_ID = 2


def create_3d_blocks(data: np.ndarray, num_blocks: int):
    total_slices = data.shape[2]
    if total_slices < num_blocks:
        return None

    slices_per_block = total_slices // num_blocks
    if slices_per_block == 0:
        return None

    blocks = []
    for block_idx in range(num_blocks):
        slice_indices = [block_idx + i * num_blocks for i in range(slices_per_block)]
        if len(slice_indices) > 10:
            effective_indices = slice_indices[1:-1]
        else:
            effective_indices = slice_indices
        blocks.append(data[:, :, effective_indices])

    return blocks


def calculate_cine_sa_metrics(cine_sa_mask_path: str, slice_num: int):
    try:
        niigz = nib.load(cine_sa_mask_path)
        pred_data = niigz.get_fdata()
        spacing = niigz.header.get_zooms()

        blocks = create_3d_blocks(pred_data, slice_num)
        if not blocks:
            return None

        valid_z_lengths = []
        for block in blocks:
            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            count = int(np.sum(has_lv))
            if count > 0:
                valid_z_lengths.append(count)

        if not valid_z_lengths:
            return None

        target_z_length = mode(valid_z_lengths, keepdims=False).mode

        block_volumes = []
        for block in blocks:
            has_lv = np.any(block == LV_BLOOD_POOL_ID, axis=(0, 1))
            valid_z_indices = np.where(has_lv)[0]

            if len(valid_z_indices) != target_z_length:
                continue

            lv_voxels = np.sum(block[..., valid_z_indices] == LV_BLOOD_POOL_ID)
            lv_vol = float(lv_voxels * spacing[0] * spacing[1] * spacing[2] / 1000.0)
            block_volumes.append(lv_vol)

        if not block_volumes:
            return None

        block_volumes.sort()
        es_lv_vol = block_volumes[0]
        ed_lv_vol = block_volumes[-1]
        lv_sv = ed_lv_vol - es_lv_vol
        lv_ef = (lv_sv / ed_lv_vol * 100.0) if ed_lv_vol > 0 else 0.0

        return {
            "LV_EDV": round(ed_lv_vol, 2),
            "LV_ESV": round(es_lv_vol, 2),
            "LV_SV": round(lv_sv, 2),
            "LV_EF": round(lv_ef, 2),
        }
    except Exception as e:
        logging.error(f"Error processing {cine_sa_mask_path}: {e}")
        return None


def convert_to_serializable(obj):
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    return obj


def main():
    parser = argparse.ArgumentParser(description="Calculate Left Ventricular Ejection Fraction (LVEF) from SAX masks.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="results",
        help="Directory containing predicted SAX segmentation masks (.nii.gz)",
    )
    parser.add_argument(
        "--slice-info",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "id_mapping.json"),
        help="Path to JSON file mapping case ID to slice count",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ef_metrics",
        help="Directory to save evaluation results",
    )
    args = parser.parse_args()

    if not os.path.exists(args.slice_info):
        print(f"Warning: Slice mapping file {args.slice_info} not found.")
        slice_info = {}
    else:
        with open(args.slice_info, "r", encoding="utf-8") as f:
            slice_info = json.load(f)

    results = []
    errors = []

    print(f"Starting calculation for SAX masks in {args.data_dir}...")
    if not os.path.isdir(args.data_dir):
        print(f"Error: Directory {args.data_dir} does not exist.")
        return

    nii_files = [f for f in sorted(os.listdir(args.data_dir)) if f.endswith(".nii.gz") or f.endswith(".nii")]
    for filename in nii_files:
        clean_name = filename.replace(".nii.gz", "").replace(".nii", "")
        case_id = None
        if clean_name[-3:] in slice_info:
            case_id = clean_name[-3:]
        elif clean_name in slice_info:
            case_id = clean_name
        else:
            for k in slice_info:
                if clean_name.endswith(k):
                    case_id = k
                    break

        if case_id is None:
            continue

        slice_num = slice_info[str(case_id)]
        mask_path = os.path.join(args.data_dir, filename)
        metrics = calculate_cine_sa_metrics(mask_path, slice_num)

        if metrics:
            res_dict = {"case_id": case_id, "file": filename, **metrics}
            results.append(res_dict)
            print(f"Case {case_id}: EDV={metrics['LV_EDV']}mL, ESV={metrics['LV_ESV']}mL, EF={metrics['LV_EF']:.2f}%")
        else:
            errors.append({"file": filename, "id": case_id, "error": "Metric computation failed"})

    os.makedirs(args.output_dir, exist_ok=True)
    out_res_path = os.path.join(args.output_dir, "lv_ef_results.json")
    out_err_path = os.path.join(args.output_dir, "lv_ef_errors.json")

    with open(out_res_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(results), f, ensure_ascii=False, indent=4)
    with open(out_err_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(errors), f, ensure_ascii=False, indent=4)

    print(f"\nCompleted: Successfully evaluated {len(results)} cases.")
    print(f"Results saved to: {out_res_path}")
    if errors:
        print(f"Errors ({len(errors)}) saved to: {out_err_path}")


if __name__ == "__main__":
    main()
