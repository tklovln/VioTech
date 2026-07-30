#!/bin/bash
#
# VioTech ablation study — training launcher
#
# Reproduces the ablations reported in the paper:
#   * Component ablation (Tables 1 & 2):
#       mert_moe              : VioTech (Ours) — full MERT Multi-Scale MoE
#       mert_no_layer_weight  : w/o MERT Layer Weight (fixed uniform layer average)
#       mert_no_cues          : w/o Transcription Cue
#       mert_no_moe           : w/o MoE (frozen MERT features + direct MLP head)
#   * Expert-count scaling (Fig. per_class_f1_vs_experts):
#       mert_moe_n2 / _n4 / _n8 / _n16 / _n32
#         (N experts, kernel/temporal window evenly spaced on the exponential
#          scale; repeat 3x with different seeds for the reported error bars.)
#
# All MERT-MoE runs use the frozen MERT-v1-330M backbone and the 8-scale
# temporal pyramid unless overridden by an expert-count case.
#
# Usage:
#   bash scripts/ablation_expert_composition_v2.sh <experiment_id> [device] [seed]
#   bash scripts/ablation_expert_composition_v2.sh all [device] [seed]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [device] [seed]}"
DEVICE="${2:-0}"
SEED="${3:-2026}"

TB="${WORKSPACE}/tb"
PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
RWC_H5="/mnt/hdd/rwc_processed_data.h5"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced.json"

# 8-scale temporal pyramid (20ms .. 2560ms), matches the paper.
TEMPORAL_EXPERTS_8="2:1,4:1,8:1,16:1,32:1,64:1,128:1,256:1"

# run_train <tag> <temporal_experts> <no_moe> <no_cues> <no_layer_weight>
run_train() {
    local TAG="$1"
    local TEMPORAL="$2"
    local NO_MOE="${3:-0}"
    local NO_CUES="${4:-0}"
    local NO_LAYER_WEIGHT="${5:-0}"

    echo ""
    echo "========================================================"
    echo "  VioTech ablation: ${TAG}"
    echo "  Temporal experts:   ${TEMPORAL}"
    echo "  w/o MoE:            ${NO_MOE}"
    echo "  w/o Transcription:  ${NO_CUES}"
    echo "  w/o Layer Weight:   ${NO_LAYER_WEIGHT}"
    echo "  Device / Seed:      ${DEVICE} / ${SEED}"
    echo "========================================================"

    local EXTRA_ARGS=""
    if [ "$NO_MOE" = "1" ]; then
        # w/o MoE: frozen MERT backbone (no expert routing) + direct MLP head.
        EXTRA_ARGS+=" --fmoe_mert_backbone=1 --fmoe_direct_head=1"
    else
        EXTRA_ARGS+=" --fmoe_mert_moe=1"
    fi
    [ "$NO_CUES" = "1" ] && EXTRA_ARGS+=" --fmoe_no_cues=1"
    [ "$NO_LAYER_WEIGHT" = "1" ] && EXTRA_ARGS+=" --fmoe_mert_no_layer_weight=1"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/main_contrast.py train \
        --workspace=$WORKSPACE \
        --logdir=$TB \
        --pretrain_path=$PRETRAIN_PATH \
        --model_tag "ablation_${TAG}" \
        --model_type='Regress_onset_offset_frame_velocity_CRNN' \
        --loss_type='regress_onset_offset_frame_velocity_bce' \
        --augmentation='aug' \
        --max_note_shift=2 \
        --batch_size=8 \
        --learning_rate=5e-4 \
        --reduce_iteration=1000 \
        --resume_iteration=0 \
        --early_stop=20000 \
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
        --fmoe_temporal_experts="${TEMPORAL}" \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --focal_gamma=2.0 \
        --mosavpt_ratio=0.2 \
        --technique_label_config="${LABEL_CFG}" \
        --rwc_eval_h5_path=$RWC_H5 \
        --rwc_eval_split=all \
        --rwc_eval_max_iterations=0 \
        $EXTRA_ARGS
}

case "$EXPERIMENT" in
    # ---- Component ablation (Tables 1 & 2) ----
    mert_moe)             run_train "mert_moe"             "$TEMPORAL_EXPERTS_8" 0 0 0 ;;
    mert_no_layer_weight) run_train "mert_no_layer_weight" "$TEMPORAL_EXPERTS_8" 0 0 1 ;;
    mert_no_cues)         run_train "mert_no_cues"         "$TEMPORAL_EXPERTS_8" 0 1 0 ;;
    mert_no_moe)          run_train "mert_no_moe"          "$TEMPORAL_EXPERTS_8" 1 0 0 ;;

    # ---- Expert-count scaling (Fig. per_class_f1_vs_experts) ----
    mert_moe_n2)  run_train "mert_moe_n2"  "2:1,256:1" 0 0 0 ;;
    mert_moe_n4)  run_train "mert_moe_n4"  "2:1,8:1,32:1,256:1" 0 0 0 ;;
    mert_moe_n8)  run_train "mert_moe_n8"  "$TEMPORAL_EXPERTS_8" 0 0 0 ;;
    mert_moe_n16) run_train "mert_moe_n16" "2:1,3:1,5:1,7:1,11:1,16:1,23:1,32:1,45:1,64:1,91:1,128:1,181:1,256:1,362:1,512:1" 0 0 0 ;;
    mert_moe_n32) run_train "mert_moe_n32" "2:1,3:1,4:1,5:1,6:1,8:1,10:1,12:1,16:1,20:1,25:1,32:1,40:1,50:1,64:1,80:1,101:1,128:1,161:1,203:1,256:1,322:1,406:1,512:1,645:1,812:1,1024:1,1290:1,1625:1,2048:1,2580:1,3250:1" 0 0 0 ;;

    all)
        for e in mert_moe mert_no_layer_weight mert_no_cues mert_no_moe \
                 mert_moe_n2 mert_moe_n4 mert_moe_n8 mert_moe_n16 mert_moe_n32; do
            bash "$0" "$e" "$DEVICE" "$SEED"
        done
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Component ablation : mert_moe mert_no_layer_weight mert_no_cues mert_no_moe"
        echo "Expert scaling     : mert_moe_n2 mert_moe_n4 mert_moe_n8 mert_moe_n16 mert_moe_n32"
        echo "All                : all"
        exit 1
        ;;
esac

echo ""
echo "Done: VioTech ablation '${EXPERIMENT}' finished."
