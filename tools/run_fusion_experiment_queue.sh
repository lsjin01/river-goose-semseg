#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON=/opt/conda/envs/goose/bin/python
export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="$REPO/third_party:$REPO"
export PYTHONUNBUFFERED=1

cd "$REPO"
mkdir -p outputs
STATUS=outputs/fusion_experiment_queue.tsv
if [[ ! -f "$STATUS" ]]; then
    printf 'timestamp\texperiment\tstatus\tconfig\n' > "$STATUS"
fi

printf '%s\tA_input_adapter\tcompleted\tconfig/labeling_data_11cls_multispectral.yaml\n' "$(date -Is)" >> "$STATUS"

experiments=(
    "B_indices:config/fusion_b_indices.yaml:fusion_b_indices"
    "C_late:config/fusion_c_late.yaml:fusion_c_late"
    "D_gated:config/fusion_d_gated.yaml:fusion_d_gated"
    "E_cross_attention:config/fusion_e_cross_attention.yaml:fusion_e_cross_attention"
)

for spec in "${experiments[@]}"; do
    IFS=: read -r experiment config run_name <<< "$spec"
    marker="outputs/$run_name/.queue_complete"
    if [[ -f "$marker" ]]; then
        printf '%s\t%s\tskipped_complete\t%s\n' "$(date -Is)" "$experiment" "$config" >> "$STATUS"
        continue
    fi

    printf '%s\t%s\trunning\t%s\n' "$(date -Is)" "$experiment" "$config" >> "$STATUS"
    args=(--config "$config" --backbone_loader hf)
    if [[ -f "outputs/$run_name/latest.pt" ]]; then
        args+=(--resume_from "outputs/$run_name/latest.pt")
    fi

    if "$PYTHON" train.py "${args[@]}"; then
        touch "$marker"
        printf '%s\t%s\tcompleted\t%s\n' "$(date -Is)" "$experiment" "$config" >> "$STATUS"
    else
        code=$?
        printf '%s\t%s\tfailed_%s\t%s\n' "$(date -Is)" "$experiment" "$code" "$config" >> "$STATUS"
        exit "$code"
    fi
done
