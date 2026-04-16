#!/bin/bash
#
# Expert Composition Ablation v2 — Fair Comparison (Unified Cues)
#
# All experiments use --fmoe_temporal_experts="11:1,51:1,51:3" to ensure
# unified cue injection (all experts receive all 3 cues: onset, offset, frame).
# This eliminates the cue-mode confound from v1.
#
# Model tags use "v2" prefix to avoid overwriting v1 checkpoints.
#
# Usage:
#   bash scripts/ablation_expert_composition_v2.sh <experiment_id> [device]
#
# Experiments:
#   baseline      : experts 0+1+2           (3 temporal, unified cues)
#   full          : experts 0+1+2+3         (+ spectral expert)
#   no_onset      : experts 1+2             (remove 110ms onset)
#   no_note       : experts 0+2             (remove 510ms note)
#   no_phrase     : experts 0+1             (remove 1530ms phrase)
#   only_onset    : expert 0 only
#   only_note     : expert 1 only
#   only_phrase   : expert 2 only
#   detach_cues   : baseline + detach onset/offset/frame cues (technique loss doesn't update transcription)
#   freeze_trans  : baseline + fully frozen transcription backbone (only technique branch trains)
#   no_moe        : baseline topology but plain MLP head instead of MoE experts (ablates MoE routing)
#   multi_crnn    : shared CNN + per-expert GRU branches (onset=fwd-1L, note=bi-2L, phrase=bi-2L)
#   hier_multilevel : hierarchical + multi-level CNN features for detector (GRU + conv2d local)
#   hier_moe      : hierarchical L1 (binary detector) + MoE L2 (4 expert fine-grained classifiers)
#   crnn_moe      : 8 heterogeneous lightweight CRNNs as experts (multi-scale temporal kernels)
#   shared_crnn_moe : shared CNN (blocks 1-3) + 8 per-expert block4 & GRU late branches
#   binary_baseline : 4 binary heads (pizz/harm/spic/legato), experts 0+1+2
#   binary_full     : 4 binary heads, experts 0+1+2+3 (+ spectral)
#   binary_no_moe   : 4 binary heads, direct MLP (no MoE routing)
#   binary_detach_cues : 4 binary heads + detach cues, experts 0+1+2
#   mert_baseline   : frozen MERT-v1-330M backbone, experts 0+1+2
#   mert_binary     : frozen MERT-v1-330M backbone + binary heads, experts 0+1+2
#   all           : run all above sequentially

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

EXPERIMENT="${1:?Usage: $0 <experiment_id> [device]}"
DEVICE="${2:-0}"

TB="${WORKSPACE}/tb"
PRETRAIN_PATH="${WORKSPACE}/checkpoints/transcriptor_model.pth"
RWC_H5="/mnt/hdd/rwc_processed_data.h5"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced.json"

SEED=2026
TEMPORAL_EXPERTS="11:1,51:1,51:3"
BINARY_LABEL_CFG="${WORKSPACE}/config/technique_label_config_binary.json"

run_train() {
    local TAG="$1"
    local EXPERT_MASK="$2"
    local SPECTRAL="${3:-0}"
    local DETACH_CUES="${4:-0}"
    local FREEZE_TRANS="${5:-0}"
    local DIRECT_HEAD="${6:-0}"
    local HIERARCHICAL="${7:-0}"
    local MULTI_CRNN="${8:-0}"
    local CRNN_MOE="${9:-0}"
    local SHARED_CRNN_MOE="${10:-0}"
    local HIER_MULTILEVEL="${11:-0}"
    local HIER_MOE="${12:-0}"
    local BINARY_TECHNIQUE="${13:-0}"
    local LABEL_CFG_OVERRIDE="${14:-}"
    local MERT_BACKBONE="${15:-0}"

    echo ""
    echo "========================================================"
    echo "  Expert Ablation v3(openstring) (unified cues): ${TAG}"
    echo "  Experts:          ${EXPERT_MASK}"
    echo "  Spectral:         ${SPECTRAL}"
    echo "  Detach cues:      ${DETACH_CUES}"
    echo "  Freeze trans:     ${FREEZE_TRANS}"
    echo "  Direct head:      ${DIRECT_HEAD}"
    echo "  Hierarchical:     ${HIERARCHICAL}"
    echo "  Multi-CRNN:       ${MULTI_CRNN}"
    echo "  CRNN-MoE:         ${CRNN_MOE}"
    echo "  Shared-CRNN-MoE:  ${SHARED_CRNN_MOE}"
    echo "  Hier-MultiLevel:  ${HIER_MULTILEVEL}"
    echo "  Hier-MoE:         ${HIER_MOE}"
    echo "  Binary technique: ${BINARY_TECHNIQUE}"
    echo "  MERT backbone:    ${MERT_BACKBONE}"
    echo "  Temporal experts: ${TEMPORAL_EXPERTS}"
    echo "  Device:           ${DEVICE}"
    echo "  Seed:             ${SEED}"
    echo "========================================================"

    local EFFECTIVE_LABEL_CFG="${LABEL_CFG}"
    [ -n "$LABEL_CFG_OVERRIDE" ] && EFFECTIVE_LABEL_CFG="$LABEL_CFG_OVERRIDE"

    local EXTRA_ARGS=""
    [ -n "$EXPERT_MASK" ] && EXTRA_ARGS+=" --fmoe_expert_mask=${EXPERT_MASK}"
    [ "$DETACH_CUES" = "1" ] && EXTRA_ARGS+=" --fmoe_detach_cues=1"
    [ "$FREEZE_TRANS" = "1" ] && EXTRA_ARGS+=" --freeze_transcription=1"
    [ "$DIRECT_HEAD" = "1" ] && EXTRA_ARGS+=" --fmoe_direct_head=1"
    [ "$HIERARCHICAL" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical=1"
    [ "$MULTI_CRNN" = "1" ] && EXTRA_ARGS+=" --fmoe_multi_crnn=1"
    [ "$CRNN_MOE" = "1" ] && EXTRA_ARGS+=" --fmoe_crnn_moe=1 --frame_moe_balance_coeff=0.005"
    [ "$SHARED_CRNN_MOE" = "1" ] && EXTRA_ARGS+=" --fmoe_shared_crnn_moe=1 --frame_moe_balance_coeff=0.005 --fmoe_aux_weight=0.1"
    [ "$HIER_MULTILEVEL" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical_multilevel=1"
    [ "$HIER_MOE" = "1" ] && EXTRA_ARGS+=" --fmoe_hierarchical_moe=1 --fmoe_aux_weight=0.1"
    [ "$BINARY_TECHNIQUE" = "1" ] && EXTRA_ARGS+=" --fmoe_binary_technique=1"
    [ "$MERT_BACKBONE" = "1" ] && EXTRA_ARGS+=" --fmoe_mert_backbone=1"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/main_contrast.py train \
        --workspace=$WORKSPACE \
        --logdir=$TB \
        --pretrain_path=$PRETRAIN_PATH \
        --model_tag "ablation_fmoe_tech0.1_${TAG}" \
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
        --fmoe_spectral_expert=${SPECTRAL} \
        --fmoe_temporal_experts="${TEMPORAL_EXPERTS}" \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --focal_gamma=2.0 \
        --mosavpt_ratio=0.2 \
        --technique_label_config="${EFFECTIVE_LABEL_CFG}" \
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
    detach_cues)  run_train "detach_cues_e012"        "0,1,2"     0  1  0  0 ;;
    freeze_trans) run_train "freeze_trans_e012"      "0,1,2"     0  0  1  0 ;;
    no_moe)       run_train "no_moe_direct"          "0,1,2"     0  0  0  1  0 ;;
    hierarchical) run_train "hierarchical"          ""          0  0  0  0  1 ;;
    multi_crnn)   run_train "multi_crnn_e012"      "0,1,2"     0  0  0  0  0  1  0 ;;
    crnn_moe)     run_train "crnn_moe_e01234567"  ""          0  0  0  0  0  0  1  0 ;;
    shared_crnn_moe) run_train "shared_crnn_moe_e01234567" "" 0  0  0  0  0  0  0  1 ;;
    hier_multilevel) run_train "hier_multilevel" ""  0  0  0  0  0  0  0  0  1  0 ;;
    hier_moe)        run_train "hier_moe"        ""  0  0  0  0  0  0  0  0  0  1 ;;
    binary_baseline)
        run_train "binary_baseline_e012" "0,1,2" 0  0  0  0  0  0  0  0  0  0  1  "$BINARY_LABEL_CFG" ;;
    binary_full)
        run_train "binary_full_e0123"    "0,1,2,3" 1  0  0  0  0  0  0  0  0  0  1  "$BINARY_LABEL_CFG" ;;
    binary_no_moe)
        run_train "binary_no_moe"        "0,1,2" 0  0  0  1  0  0  0  0  0  0  1  "$BINARY_LABEL_CFG" ;;
    binary_detach_cues)
        run_train "binary_detach_cues_e012" "0,1,2" 0  1  0  0  0  0  0  0  0  0  1  "$BINARY_LABEL_CFG" ;;
    mert_baseline)
        run_train "mert_baseline_e012" "0,1,2" 0  0  0  0  0  0  0  0  0  0  0  ""  1 ;;
    mert_binary)
        run_train "mert_binary_e012" "0,1,2" 0  0  0  0  0  0  0  0  0  0  1  "$BINARY_LABEL_CFG"  1 ;;
    all)
        for e in baseline full no_onset no_note no_phrase \
                 only_onset only_note only_phrase \
                 detach_cues freeze_trans no_moe hierarchical \
                 hier_multilevel hier_moe \
                 multi_crnn crnn_moe shared_crnn_moe \
                 binary_baseline binary_full binary_no_moe binary_detach_cues \
                 mert_baseline mert_binary; do
            bash "$0" "$e" "$DEVICE"
        done
        ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        echo "Available: baseline full no_onset no_note no_phrase"
        echo "           only_onset only_note only_phrase"
        echo "           detach_cues freeze_trans no_moe hierarchical hier_multilevel hier_moe"
        echo "           multi_crnn crnn_moe shared_crnn_moe"
        echo "           binary_baseline binary_full binary_no_moe binary_detach_cues"
        echo "           mert_baseline mert_binary all"
        exit 1
        ;;
esac

echo ""
echo "Done: expert ablation v3(openstring) '${EXPERIMENT}' finished."
