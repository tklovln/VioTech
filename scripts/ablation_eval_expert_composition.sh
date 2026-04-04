#!/bin/bash
#
# Expert Composition Ablation — Evaluation
#
# Evaluates using the best checkpoint (best_model.pth) saved during training.
# Falls back to the latest *_iterations.pth if best_model.pth not found.
#
# Usage:
#   bash scripts/ablation_eval_expert_composition.sh <experiment_id> [device]
#   bash scripts/ablation_eval_expert_composition.sh all [device]        # evaluate all
#   bash scripts/ablation_eval_expert_composition.sh <id> [device] <checkpoint>  # manual ckpt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [device] [checkpoint]}"
DEVICE="${2:-0}"
MANUAL_CKPT="${3:-}"

RWC_H5="/mnt/hdd/rwc_processed_data.h5"
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech"
SPLIT="test"

resolve_experiment() {
    local exp="$1"
    case "$exp" in
        baseline)    TAG="ablation_fmoe_baseline_e012";         EXPERT_MASK="0,1,2"; SPECTRAL=0 ;;
        full)        TAG="ablation_fmoe_full_e0123_spectral";   EXPERT_MASK="0,1,2,3"; SPECTRAL=1 ;;
        no_onset)    TAG="ablation_fmoe_no_onset_e12";          EXPERT_MASK="1,2";   SPECTRAL=0 ;;
        no_note)     TAG="ablation_fmoe_no_note_e02";           EXPERT_MASK="0,2";   SPECTRAL=0 ;;
        no_phrase)   TAG="ablation_fmoe_no_phrase_e01";         EXPERT_MASK="0,1";   SPECTRAL=0 ;;
        only_onset)  TAG="ablation_fmoe_only_onset_e0";         EXPERT_MASK="0";     SPECTRAL=0 ;;
        only_note)   TAG="ablation_fmoe_only_note_e1";          EXPERT_MASK="1";     SPECTRAL=0 ;;
        only_phrase) TAG="ablation_fmoe_only_phrase_e2";        EXPERT_MASK="2";     SPECTRAL=0 ;;
        *)
            echo "Unknown experiment: $exp"
            return 1 ;;
    esac
}

find_checkpoint() {
    local tag="$1"
    local ckpt_dir="${WORKSPACE}/checkpoints/main_contrast/${tag}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2/batch_size=4"

    # Prefer best_model.pth
    if [ -f "${ckpt_dir}/best_model.pth" ]; then
        echo "${ckpt_dir}/best_model.pth"
        return 0
    fi

    # Fallback to latest *_iterations.pth
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
    local exp="$1"
    resolve_experiment "$exp" || return 1

    local CHECKPOINT
    if [ -n "$MANUAL_CKPT" ]; then
        CHECKPOINT="$MANUAL_CKPT"
    else
        CHECKPOINT=$(find_checkpoint "$TAG")
        if [ -z "$CHECKPOINT" ]; then
            echo "[SKIP] No checkpoint found for ${exp} (${TAG})"
            return 0
        fi
    fi

    local CKPT_TYPE="latest"
    if [[ "$CHECKPOINT" == *"best_model.pth"* ]]; then
        CKPT_TYPE="best"
    fi

    echo ""
    echo "============================================"
    echo "  Expert Ablation Eval: ${exp}"
    echo "============================================"
    echo "  Tag        : ${TAG}"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "  Expert mask: ${EXPERT_MASK}"
    echo "  Spectral   : ${SPECTRAL}"
    echo "  Split      : ${SPLIT}"
    echo "  Device     : ${DEVICE}"
    echo "============================================"

    local EXTRA_ARGS=""
    [ -n "$EXPERT_MASK" ] && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"

    local RWC_ARGS=""
    [ -f "$RWC_H5" ] && RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split all"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/evaluate_frame_moe.py \
        --checkpoint_path "$CHECKPOINT" \
        --hdf5s_dir "$HDF5S_DIR" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --batch_size 4 \
        --num_workers 4 \
        --max_iterations 500 \
        --fmoe_spectral_expert=${SPECTRAL} \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        $EXTRA_ARGS \
        $RWC_ARGS

    echo ""
    echo "--------------------------------------------"
    echo "  ${exp}: done (${CKPT_TYPE} checkpoint)"
    echo "--------------------------------------------"
}

if [ "$EXPERIMENT" = "all" ]; then
    echo "Running evaluation for all expert composition experiments..."
    echo ""
    for e in baseline full no_onset no_note no_phrase \
             only_onset only_note only_phrase; do
        run_eval "$e"
    done
    echo ""
    echo "======== All expert composition evaluations complete ========"
else
    run_eval "$EXPERIMENT"
fi

'''
# 評估單一實驗（自動找 best checkpoint）
bash scripts/ablation_eval_expert_composition.sh no_onset 0
# 評估全部
bash scripts/ablation_eval_expert_composition.sh all 0
# 手動指定 checkpoint
bash scripts/ablation_eval_expert_composition.sh baseline 0 /path/to/ckpt.pth
'''