#!/usr/bin/env bash
set -uo pipefail

# STR-RepNet Clean TAR-DCR — A2_Full / LEVIR-CD-256
GPU=1
DATASET=LEVIR-CD-256
REP_MODE=full
EXP=A2_Full
EPOCHS=300
BATCH=16

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate strrep

export CUDA_VISIBLE_DEVICES=${GPU}

PROJ=/home/yqwang/projects/STR-RepNet
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run1/${EXP}/${DATASET}
LOG_DIR=/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run1/${EXP}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

for attempt in $(seq 1 200); do
    python changedetection/script/train.py \
        --cfg "${MODELS}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml" \
        --dataset "${DATASET}" \
        --dataset_root "${DS_ROOT}" \
        --train_list "${DS_ROOT}/list/train.txt" \
        --test_list "${DS_ROOT}/list/test.txt" \
        --pretrained_weight_path "${PROJ}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth" \
        --ckpt_dir "${CKPT_DIR}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH}" \
        --test_batch_size 16 \
        --crop_size 256 \
        --learning_rate 1e-4 \
        --weight_decay 5e-4 \
        --lovasz_weight 2.0 \
        --rep_mode "${REP_MODE}" \
        --num_workers 4 \
        --seed 2333 \
        --gpu 0 \
        >> "${LOG_DIR}/train_log.txt" 2>&1
    rc=$?
    if [ ${rc} -eq 0 ]; then
        echo "[DONE] training+test finished successfully" >> "${LOG_DIR}/train_log.txt"
        break
    fi
    echo "[RETRY ${attempt}] crashed with exit ${rc}, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done
