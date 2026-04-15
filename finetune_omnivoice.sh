#!/bin/bash
# ============================================================
# OmniVoice Fine-tuning — Chinese Voice Cloning Training
# ============================================================
# This script:
# 1. Exports zh-only filtered samples if needed
# 2. Splits the manifest into train/dev
# 3. Clones OmniVoice repo (for training scripts)
# 4. Tokenizes audio into WebDataset shards
# 5. Runs full fine-tuning with accelerate
#
# Prerequisites:
#   - Run download_dataset_from_hf.py to export zh-only data first
#   - Expected manifest: ./data/finetune/merged_all.jsonl
#
# Usage:
#   bash finetune_omnivoice.sh
# ============================================================

set -euo pipefail

# ====== Configuration ======
GPU_IDS="0"
NUM_GPUS=1
DEV_SYNTH_DIR="./dev_synth"
OMNIVOICE_DIR="./OmniVoice"
OUTPUT_DIR="./exp/omnivoice_finetuned"
DATA_DIR="./data/finetune"
HF_REPO_ID="${HF_REPO_ID:-amanuelbyte/omnivoice-best-of-n-training}"
TARGET_LANGUAGE="zh"
MIN_SCORE="${MIN_SCORE:-0.65}"
TRAIN_SPLIT_RATIO=0.85   # 85% train, 15% dev
# ===========================

echo "============================================================"
echo "  OmniVoice Fine-tuning Pipeline"
echo "============================================================"

# ---------------------------------------------------------------
# Step 0: Clone OmniVoice repo if needed
# ---------------------------------------------------------------
if [ ! -d "${OMNIVOICE_DIR}" ]; then
    echo "🚀 Initializing native OmniVoice architecture..."
    git clone https://github.com/k2-fsa/OmniVoice.git "${OMNIVOICE_DIR}"
    cd "${OMNIVOICE_DIR}"
    pip install -e .
    cd -
else
    echo "📦 OmniVoice repo already exists at ${OMNIVOICE_DIR}"
fi

# Fix: flex_attention crashes on GPUs with <128KB shared memory (T4, etc.).
# Run the OmniVoice patch after the repo exists so both the builder and the
# model-side BlockMask path are updated together.
python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"

export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------
# Step 1: Export zh-only data if the manifest is missing
# ---------------------------------------------------------------
echo ""
echo "📋 Step 1: Preparing zh-only manifest..."

mkdir -p "${DATA_DIR}"
MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"
DEV_JSONL="${DATA_DIR}/dev.jsonl"

if [ ! -f "${MERGED_JSONL}" ]; then
    echo "  Exporting zh-only filtered data from HF..."
    python download_dataset_from_hf.py \
        --repo-id "${HF_REPO_ID}" \
        --split train \
        --min-score "${MIN_SCORE}" \
        --languages "${TARGET_LANGUAGE}" \
        --output-dir "${DATA_DIR}/wavs" \
        --jsonl-path "${MERGED_JSONL}"
else
    echo "  Using existing manifest: ${MERGED_JSONL}"
fi

TOTAL=$(wc -l < "${MERGED_JSONL}")
echo "  Total zh samples: ${TOTAL}"

# Shuffle and split
python3 -c "
import random, sys
with open('${MERGED_JSONL}') as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * ${TRAIN_SPLIT_RATIO})
with open('${TRAIN_JSONL}', 'w') as f:
    f.writelines(lines[:split])
with open('${DEV_JSONL}', 'w') as f:
    f.writelines(lines[split:])
print(f'  Train: {split} | Dev: {len(lines) - split}')
"

# ---------------------------------------------------------------
# Step 2: Tokenize audio into WebDataset shards
# ---------------------------------------------------------------
echo ""
echo "🔧 Step 2: Tokenizing audio..."

TOKEN_DIR="${DATA_DIR}/tokens"
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

for split_name in train dev; do
    if [ "${split_name}" = "train" ]; then
        SPLIT_JSONL="${TRAIN_JSONL}"
    else
        SPLIT_JSONL="${DEV_JSONL}"
    fi

    echo "  Tokenizing ${split_name} from ${SPLIT_JSONL}..."
    CUDA_VISIBLE_DEVICES=${GPU_IDS} \
        python extract_audio_tokens_compat.py \
        --input_jsonl "${SPLIT_JSONL}" \
        --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
        --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --nj_per_gpu 1 \
        --shuffle True

    echo "  Done: ${TOKEN_DIR}/${split_name}/data.lst"
done

# ---------------------------------------------------------------
# Step 3: Create training and data config files
# ---------------------------------------------------------------
echo ""
echo "📝 Step 3: Creating config files..."

CONFIG_DIR="${DATA_DIR}/config"
mkdir -p "${CONFIG_DIR}"

# Training config — full fine-tuning, optimized for A40 48GB w/ eager attention
cat > "${CONFIG_DIR}/train_config.json" << 'TRAIN_EOF'
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,

    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.1,
    "prompt_ratio_range": [0.3, 0.7],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 1.0,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,

    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",

    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 300,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.05,
    "warmup_steps": 0,

    "batch_tokens": 3072,
    "gradient_accumulation_steps": 16,
    "num_workers": 2,

    "mixed_precision": "bf16",
    "allow_tf32": true,

    "logging_steps": 25,
    "eval_steps": 100,
    "save_steps": 100,
    "keep_last_n_checkpoints": 3
}
TRAIN_EOF

# Data config — point to tokenized shards
cat > "${CONFIG_DIR}/data_config.json" << EOF
{
    "train": [
        {
            "manifest_path": ["${TOKEN_DIR}/train/data.lst"]
        }
    ],
    "dev": [
        {
            "manifest_path": ["${TOKEN_DIR}/dev/data.lst"]
        }
    ]
}
EOF

echo "  Train config: ${CONFIG_DIR}/train_config.json"
echo "  Data config: ${CONFIG_DIR}/data_config.json"

# ---------------------------------------------------------------
# Step 4: Launch fine-tuning
# ---------------------------------------------------------------
echo ""
echo "🚀 Step 4: Starting fine-tuning..."
echo "  GPUs: ${GPU_IDS} (${NUM_GPUS})"
echo "  Output: ${OUTPUT_DIR}"

# Disable torch.compile/dynamo as a safety net — flex_attention and inductor
# can fail on GPUs with limited shared memory.  Eager works without compilation.
export TORCHDYNAMO_DISABLE=1

accelerate launch \
    --gpu_ids "${GPU_IDS}" \
    --num_processes ${NUM_GPUS} \
    train_omnivoice_lora.py \
    --lora_rank 32 \
    --train_config "${CONFIG_DIR}/train_config.json" \
    --data_config "${CONFIG_DIR}/data_config.json" \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "✅ Fine-tuning complete!"
echo "   Checkpoints saved to: ${OUTPUT_DIR}"
echo ""
echo "   To evaluate the fine-tuned model:"
echo "   python evaluation/evaluate_omnivoice.py \\"
echo "     --model-name ${OUTPUT_DIR} \\"
echo "     --whisper-lang fr --output-dir ./eval_results_omnivoice_ft/fr"
