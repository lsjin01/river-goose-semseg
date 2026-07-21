#!/usr/bin/env bash
set -euo pipefail

DEFAULT_TEACHER_CHECKPOINT="outputs/baseline_100ep/baseline_cat2_100ep/best_epoch_036_miou_0.8550.pt"

if [ "$#" -gt 2 ]; then
  echo "Usage: $0 [TEACHER_CHECKPOINT] [semantic|confidence]"
  echo
  echo "Example:"
  echo "  $0"
  echo "  $0 outputs/baseline_100ep/baseline_cat2_100ep/best_epoch_036_miou_0.8550.pt semantic"
  echo "  CUDA_VISIBLE_DEVICES=0 $0 outputs/baseline_100ep/baseline_cat2_100ep/best_epoch_036_miou_0.8550.pt confidence"
  exit 2
fi

TEACHER_CHECKPOINT="${1:-$DEFAULT_TEACHER_CHECKPOINT}"
KD_VARIANT="${2:-semantic}"
GPU_ID="${CUDA_VISIBLE_DEVICES:-1}"

if [ ! -f "$TEACHER_CHECKPOINT" ]; then
  echo "Teacher checkpoint not found: $TEACHER_CHECKPOINT" >&2
  exit 1
fi

if [ "$KD_VARIANT" != "semantic" ] && [ "$KD_VARIANT" != "confidence" ]; then
  echo "KD variant must be 'semantic' or 'confidence'. Got: $KD_VARIANT" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs/student_kd_cat2_12cls_segformer_b1_${KD_VARIANT}"
LOG_PATH="$OUTPUT_DIR/train_tqdm.log"
TMUX_SESSION="kd_cat2_segformer_b1_${KD_VARIANT}"

mkdir -p "$OUTPUT_DIR"

EXTRA_ARGS=()
if [ "$KD_VARIANT" = "confidence" ]; then
  EXTRA_ARGS+=(--enable_confidence_kd --teacher_conf_threshold 0.0)
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "tmux session already exists: $TMUX_SESSION" >&2
  echo "Attach with: tmux attach -t $TMUX_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" \
  "cd '$ROOT_DIR' && env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES='$GPU_ID' \
  conda run --no-capture-output -n goose python -u train_student_kd.py \
    --teacher_checkpoint '$TEACHER_CHECKPOINT' \
    --data_path '$ROOT_DIR/goose_data_cat2' \
    --student_model segformer_b1 \
    --num_classes 12 \
    --ignore_index 255 \
    --lambda_kd 1.0 \
    --kd_temperature 4.0 \
    --resize_width 512 \
    --resize_height 512 \
    --batch_size 12 \
    --num_workers 8 \
    --epochs 50 \
    --lr 6e-5 \
    --weight_decay 0.01 \
    --output_dir '$OUTPUT_DIR' \
    --device cuda \
    ${EXTRA_ARGS[*]} \
  2>&1 | tee -a '$LOG_PATH'"

echo "Started cat2 KD training."
echo "  session: $TMUX_SESSION"
echo "  gpu    : $GPU_ID"
echo "  teacher: $TEACHER_CHECKPOINT"
echo "  output : $OUTPUT_DIR"
echo "  log    : $LOG_PATH"
echo
echo "Attach:"
echo "  tmux attach -t $TMUX_SESSION"
echo
echo "Tail log:"
echo "  tail -f '$LOG_PATH'"
