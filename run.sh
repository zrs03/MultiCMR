#!/usr/bin/env bash
# ==============================================================================
# MultiCMR: 3D Multi-Sequence Cardiac Segmentation
# Support Single-GPU and Multi-GPU (Distributed Data Parallel via torchrun)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-train}"

# --- Default Configurations (Overridable via Environment Variables) ---
DATA_ROOT="${DATA_ROOT:-/home/liuhuan/nas/cmr_heart_models/datasets/my_dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
RESULT_DIR="${RESULT_DIR:-results}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/best_model_3d_multihead.pth}"

SOURCE_ORDER="${SOURCE_ORDER:-2ch 4ch sa}"
NUM_CLASSES_JSON="${NUM_CLASSES_JSON:-{"2ch":3,"4ch":5,"sa":4}}"
INPUT_SIZE="${INPUT_SIZE:-64 160 160}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"

# GPU Configurations
NUM_GPUS="${NUM_GPUS:-2}"          # Number of GPUs for Multi-GPU training
GPU_DEVICES="${GPU_DEVICES:-0,1}"   # GPU visible devices string, e.g. "0,1" or "0"
DEVICE="${DEVICE:-cuda:0}"         # Device for single-GPU mode

print_usage() {
  cat <<'EOF'
================================================================================
 MultiCMR Execution Script
================================================================================
Usage:
  bash run.sh train              # Run single-GPU training
  bash run.sh train-multi-gpu    # Run multi-GPU training (DDP via torchrun)
  bash run.sh test               # Run evaluation and export predictions
  bash run.sh eval-ef            # Calculate LVEF from SAX predictions

--------------------------------------------------------------------------------
Quick Command Examples:
--------------------------------------------------------------------------------
1. Single-GPU Training:
   CUDA_VISIBLE_DEVICES=0 python train.py \
     --data-root /path/to/dataset \
     --output-dir checkpoints \
     --source-order 2ch 4ch sa \
     --device cuda:0

   Or via run.sh:
   DATA_ROOT=/path/to/dataset bash run.sh train

2. Multi-GPU Training (e.g. 2 GPUs / 4 GPUs):
   CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py \
     --data-root /path/to/dataset \
     --output-dir checkpoints_ddp \
     --batch-size 1 \
     --epochs 200

   Or via run.sh:
   DATA_ROOT=/path/to/dataset NUM_GPUS=2 GPU_DEVICES=0,1 bash run.sh train-multi-gpu

3. Testing & Metric Evaluation:
   CUDA_VISIBLE_DEVICES=0 python test.py \
     --data-root /path/to/dataset \
     --checkpoint checkpoints/best_model_3d_multihead.pth \
     --output-dir results \
     --vis-per-source 3

   Or via run.sh:
   DATA_ROOT=/path/to/dataset bash run.sh test

4. LVEF Clinical Computation:
   python utils/calculate_lv_ef.py \
     --data-dir results \
     --slice-info utils/id_mapping.json \
     --output-dir results/ef_metrics

   Or via run.sh:
   bash run.sh eval-ef
================================================================================
EOF
}

if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  print_usage
  exit 0
fi

if [[ "$MODE" == "train" ]]; then
  echo "=== [MultiCMR] Starting Single-GPU Training ==="
  echo "Using Device: ${DEVICE}"
  python train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --source-order $SOURCE_ORDER \
    --image-dirname image \
    --label-dirname seg \
    --num-classes-json "$NUM_CLASSES_JSON" \
    --input-size $INPUT_SIZE \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE"

elif [[ "$MODE" == "train-multi-gpu" || "$MODE" == "train-ddp" ]]; then
  echo "=== [MultiCMR] Starting Multi-GPU DDP Training ==="
  echo "Visible GPUs: ${GPU_DEVICES} | Procs per Node: ${NUM_GPUS}"
  CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun --nproc_per_node="$NUM_GPUS" train.py \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --source-order $SOURCE_ORDER \
    --image-dirname image \
    --label-dirname seg \
    --num-classes-json "$NUM_CLASSES_JSON" \
    --input-size $INPUT_SIZE \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --epochs "$EPOCHS" \
    --lr "$LR"

elif [[ "$MODE" == "test" ]]; then
  echo "=== [MultiCMR] Starting Testing and Metric Evaluation ==="
  python test.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CHECKPOINT" \
    --output-dir "$RESULT_DIR" \
    --source-order $SOURCE_ORDER \
    --image-dirname image \
    --label-dirname seg \
    --num-classes-json "$NUM_CLASSES_JSON" \
    --input-size $INPUT_SIZE \
    --device "$DEVICE"

elif [[ "$MODE" == "eval-ef" ]]; then
  echo "=== [MultiCMR] Calculating Left Ventricular Ejection Fraction (LVEF) ==="
  python utils/calculate_lv_ef.py \
    --data-dir "${RESULT_DIR}" \
    --slice-info "utils/id_mapping.json" \
    --output-dir "${RESULT_DIR}/ef_metrics"

else
  echo "Unknown mode: $MODE"
  print_usage
  exit 1
fi
