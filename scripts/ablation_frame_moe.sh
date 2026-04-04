#!/bin/bash
#
# Frame-MoE Expert Ablation Study
#
# Usage:
#   bash scripts/ablation_frame_moe.sh <experiment_id> [device]
#
# Experiments:
#   --- Expert composition (leave-one-out & single expert) ---
#   baseline      : experts 0+1+2, top_k=2, per-task gate  (= current main model)
#   full          : experts 0+1+2+3 (add spectral)
#   no_onset      : experts 1+2   (remove onset specialist)
#   no_note       : experts 0+2   (remove note specialist)
#   no_phrase     : experts 0+1   (remove phrase specialist)
#   only_onset    : expert 0 only (single-head baseline)
#   only_note     : expert 1 only
#   only_phrase   : expert 2 only
#
#   --- Routing mechanism ---
#   topk1         : top_k=1  (most sparse: pick 1 expert per frame)
#   dense         : top_k=0  (dense softmax over all experts)
#   uniform       : equal 1/E weights, no learned gating
#   shared_gate   : single gate shared across tonal/artic/legato
#
#   --- Balance loss coefficient ---
#   bal_none      : balance_coeff=0.0
#   bal_high      : balance_coeff=0.01
#   bal_vhigh     : balance_coeff=0.1
#
#   --- Run everything ---
#   all           : run all experiments sequentially

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

# ---- Baseline defaults (match run_viotech_frame_moe_finetune.sh exactly) ----
EXPERT_MASK=""          # empty = all available (determined by spectral flag)
SPECTRAL=0
TOP_K=""                # empty = config default (2)
SHARED_GATE=0
UNIFORM=0
BALANCE=0.001
FOCAL=2.0
MOSAVPT=0.2
SEED=2026

run_train() {
    local TAG="$1"
    echo ""
    echo "========================================================"
    echo "  Ablation: ${TAG}"
    echo "  Device:   ${DEVICE}"
    echo "========================================================"

    local EXTRA_ARGS=""
    [ -n "$EXPERT_MASK" ]  && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"
    [ -n "$TOP_K" ]        && EXTRA_ARGS+=" --fmoe_top_k=${TOP_K}"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/main_contrast.py train \
        --workspace=$WORKSPACE \
        --logdir=$TB \
        --pretrain_path=$PRETRAIN_PATH \
        --model_tag "ablation_fmoe_${TAG}" \
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
        --frame_moe_balance_coeff=${BALANCE} \
        --fmoe_spectral_expert=${SPECTRAL} \
        --fmoe_shared_gate=${SHARED_GATE} \
        --fmoe_uniform_routing=${UNIFORM} \
        --focal_gamma=${FOCAL} \
        --mosavpt_ratio=${MOSAVPT} \
        --technique_label_config="${LABEL_CFG}" \
        --rwc_eval_h5_path=$RWC_H5 \
        --rwc_eval_split=all \
        --rwc_eval_max_iterations=500 \
        $EXTRA_ARGS
}

reset_defaults() {
    EXPERT_MASK=""
    SPECTRAL=0
    TOP_K=""
    SHARED_GATE=0
    UNIFORM=0
    BALANCE=0.001
}

# =========================================================================
# Experiment definitions
# =========================================================================
run_experiment() {
    local exp="$1"
    reset_defaults

    case "$exp" in
        # --- Expert composition ---
        baseline)
            EXPERT_MASK="0,1,2"
            run_train "baseline_e012"
            ;;
        full)
            EXPERT_MASK="0,1,2,3"
            SPECTRAL=1
            run_train "full_e0123_spectral"
            ;;
        no_onset)
            EXPERT_MASK="1,2"
            run_train "no_onset_e12"
            ;;
        no_note)
            EXPERT_MASK="0,2"
            run_train "no_note_e02"
            ;;
        no_phrase)
            EXPERT_MASK="0,1"
            run_train "no_phrase_e01"
            ;;
        only_onset)
            EXPERT_MASK="0"
            run_train "only_onset_e0"
            ;;
        only_note)
            EXPERT_MASK="1"
            run_train "only_note_e1"
            ;;
        only_phrase)
            EXPERT_MASK="2"
            run_train "only_phrase_e2"
            ;;

        # --- Routing mechanism ---
        topk1)
            EXPERT_MASK="0,1,2"
            TOP_K=1
            run_train "topk1_e012"
            ;;
        dense)
            EXPERT_MASK="0,1,2"
            TOP_K=0
            run_train "dense_e012"
            ;;
        uniform)
            EXPERT_MASK="0,1,2"
            UNIFORM=1
            run_train "uniform_e012"
            ;;
        shared_gate)
            EXPERT_MASK="0,1,2"
            SHARED_GATE=1
            run_train "shared_gate_e012"
            ;;

        # --- Balance loss coefficient ---
        bal_none)
            EXPERT_MASK="0,1,2"
            BALANCE=0.0
            run_train "bal0_e012"
            ;;
        bal_high)
            EXPERT_MASK="0,1,2"
            BALANCE=0.01
            run_train "bal001_e012"
            ;;
        bal_vhigh)
            EXPERT_MASK="0,1,2"
            BALANCE=0.1
            run_train "bal01_e012"
            ;;

        # --- Run all ---
        all)
            for e in baseline full \
                     no_onset no_note no_phrase \
                     only_onset only_note only_phrase \
                     topk1 dense uniform shared_gate \
                     bal_none bal_high bal_vhigh; do
                run_experiment "$e"
            done
            ;;

        *)
            echo "Unknown experiment: $exp"
            echo "Available: baseline full no_onset no_note no_phrase"
            echo "           only_onset only_note only_phrase"
            echo "           topk1 dense uniform shared_gate"
            echo "           bal_none bal_high bal_vhigh"
            echo "           all"
            exit 1
            ;;
    esac
}

run_experiment "$EXPERIMENT"
echo ""
echo "Done: ablation_fmoe experiment '${EXPERIMENT}' finished."
