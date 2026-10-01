#!/usr/bin/env bash
set -euo pipefail

# TAR-DCR Run9 C0_FTR_Post — BiFTR post-only serial reparam at stage2->3 transition, LEVIR-CD-256. GPU0 only.
GPU_ID="${1:-0}"
DATASET=LEVIR-CD-256
EXP=C0_FTR_Post

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate strrep
export CUDA_VISIBLE_DEVICES=${GPU_ID}

PROJ=/home/yqwang/projects/STR-RepNet
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run9/${EXP}/${DATASET}
LOG_DIR=/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run9/${EXP}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 200); do
    if python changedetection/script/train.py         --cfg "${MODELS}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"         --dataset "${DATASET}"         --dataset_root "${DS_ROOT}"         --train_list "${DS_ROOT}/list/train.txt"         --test_list "${DS_ROOT}/list/test.txt"         --pretrained_weight_path "${PROJ}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"         --ckpt_dir "${CKPT_DIR}"         --epochs 300         --batch_size 16         --test_batch_size 16         --crop_size 256         --learning_rate 1e-4         --weight_decay 5e-4         --lovasz_weight 2.0         --rep_mode full         --use_residual 1         --use_botr 0         --use_nscr 0         --nscr_scope high2         --encoder_train last2         --encoder_lr_ratio 0.1         --temporal_swap_prob 0.0         --decoder_dim 160         --head_mode bilinear         --use_pbru 0         --use_mpcr 0         --use_biftr 1         --biftr_mode post         --num_workers 4         --seed 2333         --gpu 0         >> "${LOG_DIR}/train_log.txt" 2>&1; then
        echo "[DONE] training+test finished successfully" >> "${LOG_DIR}/train_log.txt"
        break
    fi
    echo "[RETRY ${attempt}] crashed, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done
