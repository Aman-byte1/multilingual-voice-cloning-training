#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end VoxCPM2 LoRA finetune for zh.
# Prereq:
#   git clone https://github.com/OpenBMB/VoxCPM.git third_party/VoxCPM
#   pip install -r third_party/VoxCPM/requirements.txt

DATA_DIR="${DATA_DIR:-./data/voxcpm_zh}"
HF_DATASET="${HF_DATASET:-amanuelbyte/omnivoice-best-of-n-training}"
VOXCPM_ROOT="${VOXCPM_ROOT:-./third_party/VoxCPM}"
HF_MODEL_ID="${HF_MODEL_ID:-openbmb/VoxCPM2}"
OUT_DIR="${OUT_DIR:-./exp/voxcpm_finetuned_zh}"

python training/voxcpm/prepare_voxcpm_manifest.py \
  --dataset "${HF_DATASET}" \
  --split train \
  --output-dir "${DATA_DIR}" \
  --language-field "language" \
  --language-value "zh" \
  --text-fields "text" \
  --target-audio-fields "best_audio" \
  --ref-audio-fields "ref_audio" \
  --score-fields "best_score" \
  --min-score 0.75

python training/voxcpm/launch_voxcpm_lora.py \
  --voxcpm-root "${VOXCPM_ROOT}" \
  --hf-model-id "${HF_MODEL_ID}" \
  --train-manifest "${DATA_DIR}/train.jsonl" \
  --val-manifest "${DATA_DIR}/val.jsonl" \
  --save-path "${OUT_DIR}" \
  --num-iters 2000 \
  --max-steps 2000 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --learning-rate 1e-4 \
  --warmup-steps 100 \
  --save-interval 500 \
  --valid-interval 250 \
  --lora-rank 32 \
  --lora-alpha 16 \
  --lora-dropout 0.0
