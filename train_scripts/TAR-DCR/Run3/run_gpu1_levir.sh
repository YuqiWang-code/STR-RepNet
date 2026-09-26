#!/usr/bin/env bash
set -euo pipefail

# TAR-DCR Run3 — Edge-Basis RepDW3 (refine.dw), LEVIR only.
GPU_ID="${1:-1}"
DATASET=LEVIR-CD-256

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate strrep
export CUDA_VISIBLE_DEVICES=${GPU_ID}

PROJ=/home/yqwang/projects/STR-RepNet
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run3/${DATASET}
LOG_DIR=/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run3/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 200); do
    python changedetection/script/train.py         --cfg "${MODELS}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"         --dataset "${DATASET}"         --dataset_root "${DS_ROOT}"         --train_list "${DS_ROOT}/list/train.txt"         --test_list "${DS_ROOT}/list/test.txt"         --pretrained_weight_path "${PROJ}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"         --ckpt_dir "${CKPT_DIR}"         --epochs 300         --batch_size 16         --test_batch_size 16         --crop_size 256         --learning_rate 1e-4         --weight_decay 5e-4         --lovasz_weight 2.0         --rep_mode full         --use_residual 1         --use_edge 1         --encoder_train last2         --encoder_lr_ratio 0.1         --num_workers 4         --seed 2333         --gpu 0         >> "${LOG_DIR}/train_log.txt" 2>&1
    rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "[DONE] training+test finished successfully" >> "${LOG_DIR}/train_log.txt"
        break
    fi
    echo "[RETRY ${attempt}] crashed with exit ${rc}, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done
