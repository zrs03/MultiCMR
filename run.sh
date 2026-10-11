#!/usr/bin/env bash
# Center-window long-axis + 3D SAX segmentation.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
MODE="${1:-train}"
DATA_ROOT="${DATA_ROOT:-rawdata/CINE_MULTI}"
SPLIT_YAML="${SPLIT_YAML:-preprocess/matched.yaml}"
PREPROCESS_DIR="${PREPROCESS_DIR:-preprocess/processed}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
RESULT_DIR="${RESULT_DIR:-results}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/best_model_3d_multihead.pth}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
EPOCHS="${EPOCHS:-200}"
LR="${LR:-1e-4}"
NUM_GPUS="${NUM_GPUS:-2}"
GPU_DEVICES="${GPU_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"

# Omit size overrides by default so train/test respect YAML/checkpoint metadata.
size_args=()
if [[ -n "${LAX_SIZE:-}" ]]; then
  read -r -a lax_size <<< "$LAX_SIZE"
  size_args+=(--lax-size "${lax_size[@]}")
fi
if [[ -n "${SAX_SIZE:-}" ]]; then
  read -r -a sax_size <<< "$SAX_SIZE"
  size_args+=(--sax-size "${sax_size[@]}")
fi
train_args=(--split-yaml "$SPLIT_YAML" --output-dir "$OUTPUT_DIR"
            --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS"
            --epochs "$EPOCHS" --lr "$LR" "${size_args[@]}")
case "$MODE" in
  preprocess)
    python -m preprocess.preprocess --data-root "$DATA_ROOT" --output-yaml "$SPLIT_YAML" \
      --output-dir "$PREPROCESS_DIR" "${size_args[@]}"
    ;;
  train)
    python train.py "${train_args[@]}" --device "$DEVICE"
    ;;
  train-multi-gpu|train-ddp)
    CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun --nproc_per_node="$NUM_GPUS" \
      train.py "${train_args[@]}"
    ;;
  test)
    python test.py --split-yaml "$SPLIT_YAML" --split "${SPLIT:-val}" \
      --checkpoint "$CHECKPOINT" --output-dir "$RESULT_DIR" --device "$DEVICE" \
      --eval-mode "${EVAL_MODE:-both}" "${size_args[@]}"
    ;;
  eval-ef)
    # This utility requires predictions restored to the original SAX layout/spacing.
    python utils/calculate_lv_ef.py --data-dir "${EF_DATA_DIR:?Set EF_DATA_DIR to original-space SAX masks}" \
      --slice-info "${SLICE_INFO:-rawdata/id_mapping.json}" --output-dir "$RESULT_DIR/ef_metrics"
    ;;
  -h|--help)
    cat <<'EOF'
Usage: bash run.sh {preprocess|train|train-multi-gpu|test|eval-ef}
  preprocess: cache 2CH/4CH [5,128,128] windows, center labels [128,128], SAX [128,128,128] (D,H,W)
  train: paired LAX windows + shared SAX; center-frame supervision
  test: sliding windows reconstruct native LAX frames; average SAX probabilities; default split=val
Environment: DATA_ROOT, SPLIT_YAML, PREPROCESS_DIR, OUTPUT_DIR, RESULT_DIR,
  CHECKPOINT, DEVICE, BATCH_SIZE, EPOCHS, LR, NUM_WORKERS, NUM_GPUS, GPU_DEVICES,
  LAX_SIZE, SAX_SIZE, SPLIT, EVAL_MODE.
Examples:
  bash run.sh preprocess
  DEVICE=cuda:0 bash run.sh train
  SPLIT=test EVAL_MODE=all-seven bash run.sh test
  EF_DATA_DIR=/path/to/original_space_masks bash run.sh eval-ef
EOF
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 1 ;;
esac
