#!/bin/bash
#
# Temporal Density & Cue Ablation — Training
#
# Tests (A) varying number of temporal experts at different granularities,
# (B) with vs without transcription cue injection.
# All temporal experts use unified "all cues" injection (when cues enabled).
#
# Usage:
#   bash scripts/ablation_temporal_density.sh <experiment_id> [device]
#
# Experiments:
#   --- Granularity (temporal range & density) ---
#   t3_baseline   : 3 experts — 110ms, 510ms, 1.5s       (current default)
#   t3_wide       : 3 experts — 50ms, 510ms, 3.5s         (wider range, same count)
#   t5_fine       : 5 experts — 110ms, 250ms, 510ms, 1.0s, 1.5s
#   t5_wide       : 5 experts — 50ms, 250ms, 510ms, 1.5s, 3.5s
#   t7_full       : 7 experts — 50ms, 110ms, 250ms, 510ms, 1.0s, 1.5s, 2.5s
#
#   --- Cue ablation (with t3_baseline config) ---
#   t3_no_cues    : 3 experts, NO transcription cue injection
#
#   --- Combined ---
#   t5_no_cues    : 5 experts (fine), NO cues
#
#   all           : run all above

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [device]}"
DEVICE="${2:-0}"

if [ ! -d "${WORKSPACE}/hdf5s/viotech" ]; then
    mkdir -p "${WORKSPACE}/hdf5s"
    echo "Creating symlink for viotech dataset..."
    ln -s /mnt/hdd/viotech "${WORKSPACE}/hdf5s/viotech"
fi

TB="${WORKSPACE}/tb"
PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
RWC_H5="/mnt/hdd/rwc_processed_data.h5"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced.json"

SEED=2026

run_train() {
    local TAG="$1"
    local TEMPORAL="$2"
    local NO_CUES="${3:-0}"

    echo ""
    echo "========================================================"
    echo "  Temporal Density Ablation: ${TAG}"
    echo "  Temporal experts: ${TEMPORAL}"
    echo "  Cues disabled:   ${NO_CUES}"
    echo "  Device:          ${DEVICE}"
    echo "  Seed:            ${SEED}"
    echo "========================================================"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/main_contrast.py train \
        --workspace=$WORKSPACE \
        --logdir=$TB \
        --pretrain_path=$PRETRAIN_PATH \
        --model_tag "ablation_tdensity_${TAG}" \
        --model_type='Regress_onset_offset_frame_velocity_CRNN' \
        --loss_type='regress_onset_offset_frame_velocity_bce' \
        --augmentation='aug' \
        --max_note_shift=2 \
        --batch_size=4 \
        --learning_rate=5e-4 \
        --reduce_iteration=1000 \
        --resume_iteration=0 \
        --early_stop=10000 \
        --patience=10000 \
        --best_metric=fmoe_loss_technique \
        --device $DEVICE \
        --seed $SEED \
        --dataset viotech_mixed_mosavpt \
        --contrast_weight=0.0 \
        --ctc_weight=0.0 \
        --num_workers=8 \
        --technique_weight=0.0 \
        --technique_moe_weight=0.0 \
        --technique_moe_zone_weight=0.0 \
        --technique_moe_zone_pt_weight=0.0 \
        --technique_frame_moe_weight=1 \
        --frame_moe_balance_coeff=0.001 \
        --fmoe_spectral_expert=0 \
        --fmoe_top_k=0 \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --fmoe_temporal_experts="${TEMPORAL}" \
        --fmoe_no_cues=${NO_CUES} \
        --focal_gamma=2.0 \
        --mosavpt_ratio=0.2 \
        --technique_label_config="${LABEL_CFG}" \
        --rwc_eval_h5_path=$RWC_H5 \
        --rwc_eval_split=all \
        --rwc_eval_max_iterations=500
}

# Temporal expert definitions (kernel:dilation)
#   5:1=50ms  11:1=110ms  25:1=250ms  51:1=510ms
#   51:2=1.0s  51:3=1.5s  51:5=2.5s  51:7=3.5s

case "$EXPERIMENT" in
    # --- Granularity ---
    t3_baseline)
        run_train "t3_baseline" "11:1,51:1,51:3" 0
        ;;
    t3_wide)
        run_train "t3_wide" "5:1,51:1,51:7" 0
        ;;
    t5_fine)
        run_train "t5_fine" "11:1,25:1,51:1,51:2,51:3" 0
        ;;
    t5_wide)
        run_train "t5_wide" "5:1,25:1,51:1,51:3,51:7" 0
        ;;
    t7_full)
        run_train "t7_full" "5:1,11:1,25:1,51:1,51:2,51:3,51:5" 0
        ;;

    # --- Cue ablation ---
    t3_no_cues)
        run_train "t3_no_cues" "11:1,51:1,51:3" 1
        ;;
    t5_no_cues)
        run_train "t5_no_cues" "11:1,25:1,51:1,51:2,51:3" 1
        ;;

    # --- All ---
    all)
        for e in t3_baseline t3_wide t5_fine t5_wide t7_full \
                 t3_no_cues t5_no_cues; do
            bash "$0" "$e" "$DEVICE"
        done
        ;;

    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: t3_baseline t3_wide t5_fine t5_wide t7_full"
        echo "           t3_no_cues t5_no_cues all"
        exit 1
        ;;
esac

echo ""
echo "Done: temporal density ablation '${EXPERIMENT}' finished."
