#!/bin/bash
#
# Temporal Density & Cue Ablation — Evaluation
#
# Evaluates trained models from ablation_temporal_density.sh
# Prefers best_model.pth; falls back to final checkpoint.
# Prints checkpoint info, saves per-experiment JSON, and summary table.
#
# Usage:
#   bash scripts/ablation_eval_temporal_density.sh <experiment_id> [device]
#
# experiment_id: t3_baseline t3_wide t5_fine t5_wide t7_full
#                t3_no_cues t5_no_cues
#                all (evaluate every experiment)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [device]}"
DEVICE="${2:-0}"

RWC_H5="/mnt/hdd/rwc_processed_data.h5"
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech"
SPLIT="validation"
RESULTS_DIR="${WORKSPACE}/results/ablation_temporal_density"
mkdir -p "$RESULTS_DIR"

find_checkpoint() {
    local tag="$1"
    local ckpt_dir="${WORKSPACE}/checkpoints/main_contrast/${tag}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4"

    if [ -f "${ckpt_dir}/best_model.pth" ]; then
        echo "${ckpt_dir}/best_model.pth"
        return 0
    fi

    local latest
    latest=$(ls -t "${ckpt_dir}"/*_iterations.pth 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "$latest"
        return 0
    fi

    echo ""
    return 1
}

run_eval() {
    local TAG="$1"
    local TEMPORAL="$2"
    local NO_CUES="${3:-0}"

    local MODEL_TAG="ablation_tdensity_${TAG}"

    local CHECKPOINT
    CHECKPOINT=$(find_checkpoint "$MODEL_TAG")
    if [ -z "$CHECKPOINT" ]; then
        echo "[SKIP] No checkpoint found for ${TAG} (${MODEL_TAG})"
        return 0
    fi

    local CKPT_TYPE="latest"
    if [[ "$CHECKPOINT" == *"best_model.pth"* ]]; then
        CKPT_TYPE="best"
    fi

    echo ""
    echo "========================================================"
    echo "  Temporal Density Eval: ${TAG}"
    echo "========================================================"
    echo "  Tag        : ${MODEL_TAG}"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "  Temporal   : ${TEMPORAL}"
    echo "  No cues    : ${NO_CUES}"
    echo "  Split      : ${SPLIT}"
    echo "  Device     : ${DEVICE}"
    echo "========================================================"

    # Print best checkpoint info
    python3 -c "
import torch
ckpt = torch.load('${CHECKPOINT}', map_location='cpu', weights_only=False)
it = ckpt.get('iteration', 'N/A')
bm = ckpt.get('best_metric', 'N/A')
bv = ckpt.get('best_metric_val', None)
bv_str = f'{bv:.6f}' if bv is not None else 'N/A'
print(f'  Best step     : {it}')
print(f'  Best metric   : {bm} = {bv_str}')
"

    local RWC_ARGS=""
    [ -f "$RWC_H5" ] && RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split all"

    local JSON_OUT="${RESULTS_DIR}/${TAG}.json"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/evaluate_frame_moe.py \
        --checkpoint_path "${CHECKPOINT}" \
        --hdf5s_dir "$HDF5S_DIR" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --batch_size 4 \
        --num_workers 4 \
        --max_iterations 500 \
        --fmoe_spectral_expert=0 \
        --fmoe_temporal_experts="${TEMPORAL}" \
        --fmoe_no_cues=${NO_CUES} \
        --output_json "$JSON_OUT" \
        $RWC_ARGS

    echo ""
    echo "--------------------------------------------"
    echo "  ${TAG}: done (${CKPT_TYPE} checkpoint)"
    echo "  Results → ${JSON_OUT}"
    echo "--------------------------------------------"
}

case "$EXPERIMENT" in
    t3_baseline) run_eval "t3_baseline" "11:1,51:1,51:3" 0 ;;
    t3_wide)     run_eval "t3_wide"     "5:1,51:1,51:7" 0 ;;
    t5_fine)     run_eval "t5_fine"     "11:1,25:1,51:1,51:2,51:3" 0 ;;
    t5_wide)     run_eval "t5_wide"     "5:1,25:1,51:1,51:3,51:7" 0 ;;
    t7_full)     run_eval "t7_full"     "5:1,11:1,25:1,51:1,51:2,51:3,51:5" 0 ;;
    t3_no_cues)  run_eval "t3_no_cues"  "11:1,51:1,51:3" 1 ;;
    t5_no_cues)  run_eval "t5_no_cues"  "11:1,25:1,51:1,51:2,51:3" 1 ;;

    all)
        for e in t3_baseline t3_wide t5_fine t5_wide t7_full \
                 t3_no_cues t5_no_cues; do
            bash "$0" "$e" "$DEVICE"
        done

        # Merge individual JSONs into a summary
        SUMMARY="${RESULTS_DIR}/summary.json"
        python3 -c "
import json, glob, os
results = {}
for f in sorted(glob.glob('${RESULTS_DIR}/*.json')):
    name = os.path.splitext(os.path.basename(f))[0]
    if name == 'summary':
        continue
    with open(f) as fp:
        results[name] = json.load(fp)
with open('${SUMMARY}', 'w') as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print(f'Summary saved to ${SUMMARY}')
print()
header = f'{\"Experiment\":<15s} {\"Step\":>6s}  {\"Tonal Acc\":>9s} {\"Tonal F1\":>9s} {\"Artic Acc\":>9s} {\"Artic F1\":>9s} {\"Legato Acc\":>10s} {\"Legato F1\":>10s}'
print(header)
print('-' * len(header))
for name, data in results.items():
    step = data.get('checkpoint', {}).get('iteration', '?')
    vio = data.get('viotech', {})
    t = vio.get('tonal', {})
    a = vio.get('articulation', {})
    l = vio.get('legato', {})
    print(f'{name:<15s} {str(step):>6s}  {t.get(\"accuracy\",0):9.4f} {t.get(\"macro_f1\",0):9.4f} {a.get(\"accuracy\",0):9.4f} {a.get(\"macro_f1\",0):9.4f} {l.get(\"accuracy\",0):10.4f} {l.get(\"macro_f1\",0):10.4f}')
"

        echo ""
        echo "======== All temporal density evaluations complete ========"
        echo "Results dir: ${RESULTS_DIR}/"
        ;;

    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: t3_baseline t3_wide t5_fine t5_wide t7_full"
        echo "           t3_no_cues t5_no_cues all"
        exit 1
        ;;
esac

echo ""
echo "All evaluations for '${EXPERIMENT}' complete."
