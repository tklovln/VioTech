#!/bin/bash
#
# Expert Composition Ablation — Training
#
# Focused ablation on which experts matter. Saves best checkpoint
# based on fmoe_loss_technique (lower is better).
#
# Usage:
#   bash scripts/ablation_expert_composition.sh <experiment_id> [device]
#
# Experiments:
#   baseline      : experts 0+1+2           (current main model)
#   full          : experts 0+1+2+3         (add spectral expert)
#   no_onset      : experts 1+2             (remove onset specialist)
#   no_note       : experts 0+2             (remove note specialist)
#   no_phrase     : experts 0+1             (remove phrase specialist)
#   only_onset    : expert 0 only
#   only_note     : expert 1 only
#   only_phrase   : expert 2 only
#   all           : run all above sequentially

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

SEED=42

run_train() {
    local TAG="$1"
    local EXPERT_MASK="$2"
    local SPECTRAL="${3:-0}"

    echo ""
    echo "========================================================"
    echo "  Expert Ablation: ${TAG}"
    echo "  Experts:  ${EXPERT_MASK}"
    echo "  Spectral: ${SPECTRAL}"
    echo "  Device:   ${DEVICE}"
    echo "  Seed:     ${SEED}"
    echo "========================================================"

    local EXTRA_ARGS=""
    [ -n "$EXPERT_MASK" ] && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"

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
        --frame_moe_balance_coeff=0.001 \
        --fmoe_spectral_expert=${SPECTRAL} \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --focal_gamma=2.0 \
        --mosavpt_ratio=0.2 \
        --technique_label_config="${LABEL_CFG}" \
        --rwc_eval_h5_path=$RWC_H5 \
        --rwc_eval_split=all \
        --rwc_eval_max_iterations=500 \
        $EXTRA_ARGS
}

case "$EXPERIMENT" in
    baseline)    run_train "baseline_e012"         "0,1,2"     0 ;;
    full)        run_train "full_e0123_spectral"   "0,1,2,3"   1 ;;
    no_onset)    run_train "no_onset_e12"           "1,2"       0 ;;
    no_note)     run_train "no_note_e02"            "0,2"       0 ;;
    no_phrase)   run_train "no_phrase_e01"           "0,1"       0 ;;
    only_onset)  run_train "only_onset_e0"           "0"         0 ;;
    only_note)   run_train "only_note_e1"            "1"         0 ;;
    only_phrase) run_train "only_phrase_e2"           "2"         0 ;;
    all)
        for e in baseline full no_onset no_note no_phrase \
                 only_onset only_note only_phrase; do
            bash "$0" "$e" "$DEVICE"
        done
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: baseline full no_onset no_note no_phrase"
        echo "           only_onset only_note only_phrase all"
        exit 1
        ;;
esac

echo ""
echo "Done: expert ablation '${EXPERIMENT}' finished."
