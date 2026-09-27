#!/usr/bin/env bash
set -euo pipefail

# TAR-DCR Run5 D0 — full-encoder upper-bound diagnostic (LEVIR only).
# Dual-GPU memory gate: waits until GPU0 OR GPU1 has >= 16 GiB free (shared
# server, CASA-CD co-tenant), then starts on whichever GPU freed up first.
DATASET=LEVIR-CD-256
EXP=D0_full_encoder

source /home/yqwang/miniforge3/etc/profile.d/conda.sh
conda activate strrep

PROJ=/home/yqwang/projects/STR-RepNet
MODELS=${PROJ}/models
DS_ROOT=/share_datasets/CD/${DATASET}
CKPT_DIR=/share_datasets/yqwang/checkpoints/STR-RepNet/TAR-DCR/Run5/${EXP}/${DATASET}
LOG_DIR=/home/yqwang/outputs/STR-RepNet/TAR-DCR/Run5/${EXP}/${DATASET}

mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
cd "${MODELS}"

# wait for >= 16 GiB free on either GPU (max ~36h)
PICK=""
for _wait in $(seq 1 432); do
    FREE0=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    FREE1=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${FREE0:-0}" -ge 16000 ]; then
        PICK=0
        break
    fi
    if [ "${FREE1:-0}" -ge 16000 ]; then
        PICK=1
        break
    fi
    echo "[WAIT-MEM] gpu0=${FREE0} gpu1=${FREE1} MiB (<16000), sleeping 300s" >> "${LOG_DIR}/train_log.txt"
    sleep 300
done
echo "[WAIT-MEM] starting on gpu${PICK}" >> "${LOG_DIR}/train_log.txt"
export CUDA_VISIBLE_DEVICES=${PICK}

for attempt in $(seq 1 200); do
    if python changedetection/script/train.py         --cfg "${MODELS}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"         --dataset "${DATASET}"         --dataset_root "${DS_ROOT}"         --train_list "${DS_ROOT}/list/train.txt"         --test_list "${DS_ROOT}/list/test.txt"         --pretrained_weight_path "${PROJ}/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth"         --ckpt_dir "${CKPT_DIR}"         --epochs 300         --batch_size 16         --test_batch_size 16         --crop_size 256         --learning_rate 1e-4         --weight_decay 5e-4         --lovasz_weight 2.0         --rep_mode full         --use_residual 1         --use_botr 0         --encoder_train full         --encoder_lr_ratio 0.1         --temporal_swap_prob 0.0         --num_workers 4         --seed 2333         --gpu 0         >> "${LOG_DIR}/train_log.txt" 2>&1; then
        echo "[DONE] training+test finished successfully" >> "${LOG_DIR}/train_log.txt"
        break
    fi
    echo "[RETRY ${attempt}] crashed, resuming from last.pth in 10s" >> "${LOG_DIR}/train_log.txt"
    sleep 10
done
