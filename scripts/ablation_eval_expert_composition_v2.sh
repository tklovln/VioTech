#!/bin/bash
#
# VioTech ablation study — evaluation launcher
#
# Evaluates the checkpoints produced by ablation_expert_composition_v2.sh and
# (optionally) generates the UMAP embedding plots. Cases mirror the paper:
#   Component ablation : mert_moe mert_no_layer_weight mert_no_cues mert_no_moe
#   Expert scaling     : mert_moe_n2 mert_moe_n4 mert_moe_n8 mert_moe_n16 mert_moe_n32
#
# Usage:
#   bash scripts/ablation_eval_expert_composition_v2.sh <experiment_id> [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh all [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh --umap <experiment_id> [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh --umap all [device]
#   bash scripts/ablation_eval_expert_composition_v2.sh --iter <N> <experiment_id> [device]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKSPACE="${PROJECT_ROOT}"

UMAP_ONLY=0
ITER_OVERRIDE=""
while true; do
    case "${1:-}" in
        --umap) UMAP_ONLY=1; shift ;;
        --iter) ITER_OVERRIDE="${2:?--iter requires an iteration number}"; shift 2 ;;
        *) break ;;
    esac
done

EXPERIMENT="${1:?Usage: $0 [--umap] <experiment_id> [device] [checkpoint]}"
DEVICE="${2:-0}"
MANUAL_CKPT="${3:-}"

RWC_H5="/mnt/hdd/rwc_processed_data.h5"
HDF5S_DIR="${WORKSPACE}/hdf5s/viotech_paganini_kayser_with_openstring"
LABEL_CFG="${WORKSPACE}/config/technique_label_config_reduced.json"
SPLIT="test"
RESULTS_DIR="${WORKSPACE}/results/ablation_expert_composition_v2"
TEMPORAL_EXPERTS_8="2:1,4:1,8:1,16:1,32:1,64:1,128:1,256:1"
mkdir -p "$RESULTS_DIR"

# Sets: TAG, TEMPORAL, NO_MOE, NO_CUES, NO_LAYER_WEIGHT
resolve_experiment() {
    local exp="$1"
    NO_MOE=0; NO_CUES=0; NO_LAYER_WEIGHT=0
    TEMPORAL="$TEMPORAL_EXPERTS_8"
    case "$exp" in
        mert_moe)             TAG="ablation_mert_moe" ;;
        mert_no_layer_weight) TAG="ablation_mert_no_layer_weight"; NO_LAYER_WEIGHT=1 ;;
        mert_no_cues)         TAG="ablation_mert_no_cues"; NO_CUES=1 ;;
        mert_no_moe)          TAG="ablation_mert_no_moe"; NO_MOE=1 ;;
        mert_moe_n2)  TAG="ablation_mert_moe_n2";  TEMPORAL="2:1,256:1" ;;
        mert_moe_n4)  TAG="ablation_mert_moe_n4";  TEMPORAL="2:1,8:1,32:1,256:1" ;;
        mert_moe_n8)  TAG="ablation_mert_moe_n8";  TEMPORAL="$TEMPORAL_EXPERTS_8" ;;
        mert_moe_n16) TAG="ablation_mert_moe_n16"; TEMPORAL="2:1,3:1,5:1,7:1,11:1,16:1,23:1,32:1,45:1,64:1,91:1,128:1,181:1,256:1,362:1,512:1" ;;
        mert_moe_n32) TAG="ablation_mert_moe_n32"; TEMPORAL="2:1,3:1,4:1,5:1,6:1,8:1,10:1,12:1,16:1,20:1,25:1,32:1,40:1,50:1,64:1,80:1,101:1,128:1,161:1,203:1,256:1,322:1,406:1,512:1,645:1,812:1,1024:1,1290:1,1625:1,2048:1,2580:1,3250:1" ;;
        *)
            echo "Unknown experiment: $exp"
            return 1 ;;
    esac
}

find_checkpoint() {
    local tag="$1"
    local base="${WORKSPACE}/checkpoints/main_contrast/${tag}/Regress_onset_offset_frame_velocity_CRNN/loss_type=regress_onset_offset_frame_velocity_bce/augmentation=aug/max_note_shift=2"
    local ckpt_dir
    for bs in 4 8; do
        ckpt_dir="${base}/batch_size=${bs}"
        if [ -n "$ITER_OVERRIDE" ]; then
            if [ -f "${ckpt_dir}/${ITER_OVERRIDE}_iterations.pth" ]; then
                echo "${ckpt_dir}/${ITER_OVERRIDE}_iterations.pth"; return 0
            fi
            continue
        fi
        if [ -f "${ckpt_dir}/best_model.pth" ]; then
            echo "${ckpt_dir}/best_model.pth"; return 0
        fi
        local latest
        latest=$(ls -t "${ckpt_dir}"/*_iterations.pth 2>/dev/null | head -1)
        if [ -n "$latest" ]; then echo "$latest"; return 0; fi
    done
    echo ""; return 1
}

_resolve_checkpoint_and_args() {
    local exp="$1"
    resolve_experiment "$exp" || return 1

    if [ -n "$MANUAL_CKPT" ]; then
        CHECKPOINT="$MANUAL_CKPT"
    else
        CHECKPOINT=$(find_checkpoint "$TAG")
        if [ -z "$CHECKPOINT" ]; then
            echo "[SKIP] No checkpoint found for ${exp} (${TAG})"
            return 1
        fi
    fi

    CKPT_TYPE="latest"
    if [[ "$CHECKPOINT" == *"best_model.pth"* ]]; then
        CKPT_TYPE="best"
    elif [ -n "$ITER_OVERRIDE" ]; then
        CKPT_TYPE="iter${ITER_OVERRIDE}"
    fi

    EXTRA_ARGS=""
    if [ "$NO_MOE" = "1" ]; then
        EXTRA_ARGS+=" --fmoe_mert_backbone=1 --fmoe_direct_head=1"
    else
        EXTRA_ARGS+=" --fmoe_mert_moe=1"
    fi
    [ "$NO_CUES" = "1" ] && EXTRA_ARGS+=" --fmoe_no_cues=1"
    [ "$NO_LAYER_WEIGHT" = "1" ] && EXTRA_ARGS+=" --fmoe_mert_no_layer_weight=1"

    RWC_ARGS=""
    [ -f "$RWC_H5" ] && RWC_ARGS="--rwc_h5_path $RWC_H5 --rwc_split all"
}

run_umap() {
    local exp="$1"
    _resolve_checkpoint_and_args "$exp" || return 0

    echo ""
    echo "============================================"
    echo "  UMAP only: ${exp}"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "============================================"

    cd "${WORKSPACE}/piano_transcription"
    local UMAP_DIR="${RESULTS_DIR}/figures"
    mkdir -p "$UMAP_DIR"

    python3 pytorch/visualize_umap_frame_moe.py \
        --checkpoint_path "$CHECKPOINT" \
        --hdf5s_dir "$HDF5S_DIR" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --batch_size 4 \
        --num_workers 4 \
        --fmoe_spectral_expert=0 \
        --fmoe_temporal_experts="${TEMPORAL}" \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --technique_label_config="${LABEL_CFG}" \
        --output_dir "$UMAP_DIR" \
        --tag "$exp" \
        --n_per_class 1000 \
        --max_batches 0 \
        $EXTRA_ARGS \
        $RWC_ARGS

    echo "  ${exp}: UMAP → ${UMAP_DIR}/umap_${exp}_*.png"
}

run_eval() {
    local exp="$1"
    _resolve_checkpoint_and_args "$exp" || return 0

    local JSON_SUFFIX=""
    [ -n "$ITER_OVERRIDE" ] && JSON_SUFFIX="_iter${ITER_OVERRIDE}"

    echo ""
    echo "============================================"
    echo "  VioTech ablation eval: ${exp}"
    echo "  Tag        : ${TAG}"
    echo "  Checkpoint : ${CHECKPOINT} (${CKPT_TYPE})"
    echo "  Temporal   : ${TEMPORAL}"
    echo "  Split      : ${SPLIT}"
    echo "  Device     : ${DEVICE}"
    echo "============================================"

    python3 -c "
import torch
ckpt = torch.load('${CHECKPOINT}', map_location='cpu', weights_only=False)
it = ckpt.get('iteration', 'N/A')
bm = ckpt.get('best_metric', 'N/A')
bv = ckpt.get('best_metric_val', None)
bv_str = f'{bv:.6f}' if bv is not None else 'N/A'
print(f'  Best step   : {it}')
print(f'  Best metric : {bm} = {bv_str}')
"

    local JSON_OUT="${RESULTS_DIR}/${exp}${JSON_SUFFIX}.json"

    cd "${WORKSPACE}/piano_transcription"

    python3 pytorch/evaluate_frame_moe.py \
        --checkpoint_path "$CHECKPOINT" \
        --hdf5s_dir "$HDF5S_DIR" \
        --split "$SPLIT" \
        --device "$DEVICE" \
        --batch_size 4 \
        --num_workers 4 \
        --max_iterations 0 \
        --fmoe_spectral_expert=0 \
        --fmoe_temporal_experts="${TEMPORAL}" \
        --fmoe_shared_gate=0 \
        --fmoe_uniform_routing=0 \
        --technique_label_config="${LABEL_CFG}" \
        --output_json "$JSON_OUT" \
        $EXTRA_ARGS \
        $RWC_ARGS

    echo "  ${exp}: done (${CKPT_TYPE}). Results → ${JSON_OUT}"
}

ALL_EXPERIMENTS="mert_moe mert_no_layer_weight mert_no_cues mert_no_moe \
    mert_moe_n2 mert_moe_n4 mert_moe_n8 mert_moe_n16 mert_moe_n32"

if [ "$UMAP_ONLY" = "1" ]; then
    if [ "$EXPERIMENT" = "all" ]; then
        for e in $ALL_EXPERIMENTS; do run_umap "$e"; done
    else
        run_umap "$EXPERIMENT"
    fi
    echo ""
    echo "======== UMAP generation complete ========"
    exit 0
fi

if [ "$EXPERIMENT" = "all" ]; then
    JSON_SUFFIX=""
    [ -n "$ITER_OVERRIDE" ] && JSON_SUFFIX="_iter${ITER_OVERRIDE}"
    for e in $ALL_EXPERIMENTS; do run_eval "$e"; done

    SUMMARY="${RESULTS_DIR}/summary${JSON_SUFFIX}.json"
    python3 -c "
import json, glob, os
iter_override = '${ITER_OVERRIDE}'
suffix = ('_iter' + iter_override) if iter_override else ''
results = {}
for f in sorted(glob.glob('${RESULTS_DIR}/*' + suffix + '.json')):
    name = os.path.splitext(os.path.basename(f))[0]
    if name.startswith('summary'):
        continue
    if not iter_override and ('_iter' in name):
        continue
    with open(f) as fp:
        results[name] = json.load(fp)
with open('${SUMMARY}', 'w') as fp:
    json.dump(results, fp, indent=2, ensure_ascii=False)
print(f'Summary saved to ${SUMMARY}')
print()
header = f'{\"Experiment\":<22s} {\"Step\":>6s}  {\"Tonal Acc\":>9s} {\"Tonal F1\":>9s} {\"Artic Acc\":>9s} {\"Artic F1\":>9s} {\"Legato Acc\":>10s} {\"Legato F1\":>10s}'
print(header)
print('-' * len(header))
for name, data in results.items():
    step = data.get('checkpoint', {}).get('iteration', '?')
    vio = data.get('viotech', {})
    t = vio.get('tonal', {}); a = vio.get('articulation', {}); l = vio.get('legato', {})
    print(f'{name:<22s} {str(step):>6s}  {t.get(\"accuracy\",0):9.4f} {t.get(\"macro_f1\",0):9.4f} {a.get(\"accuracy\",0):9.4f} {a.get(\"macro_f1\",0):9.4f} {l.get(\"accuracy\",0):10.4f} {l.get(\"macro_f1\",0):10.4f}')
"
    echo ""
    echo "======== All ablation evaluations complete ========"
    echo "Results dir: ${RESULTS_DIR}/"
else
    run_eval "$EXPERIMENT"
fi
