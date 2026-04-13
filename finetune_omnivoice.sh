#!/bin/bash
# ============================================================
# OmniVoice Fine-tuning — Best-of-N Ensemble Training
# ============================================================
# This script:
# 1. Merges per-language JSONL manifests into train/dev splits
# 2. Clones OmniVoice repo (for training scripts)
# 3. Tokenizes audio into WebDataset shards
# 4. Runs full fine-tuning with accelerate
#
# Prerequisites:
#   - Run synthesize_dev_best_of_n.py for all 3 languages first
#   - Outputs: dev_synth/train_fr.jsonl, train_ar.jsonl, train_zh.jsonl
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
TRAIN_SPLIT_RATIO=0.85   # 85% train, 15% dev
# ===========================

echo "============================================================"
echo "  OmniVoice Fine-tuning Pipeline"
echo "============================================================"

# ---------------------------------------------------------------
# Step 0: Clone OmniVoice repo if needed
# ---------------------------------------------------------------
if [ ! -d "${OMNIVOICE_DIR}" ]; then
    echo "📦 Cloning OmniVoice repository..."
    git clone https://github.com/k2-fsa/OmniVoice.git "${OMNIVOICE_DIR}"
    cd "${OMNIVOICE_DIR}"
    pip install -e .
    cd -
else
    echo "📦 OmniVoice repo already exists at ${OMNIVOICE_DIR}"
fi

# Fix: flex_attention crashes on GPUs with <128KB shared memory (T4, etc.).
# The OmniVoice builder hardcodes flex_attention which requires torch.compile +
# Triton autotuning.  Switch to SDPA which works on all GPUs.
if grep -q 'flex_attention' "${OMNIVOICE_DIR}/omnivoice/training/builder.py" 2>/dev/null; then
    sed -i 's/attn_implementation="flex_attention"/attn_implementation="eager"/g' \
        "${OMNIVOICE_DIR}/omnivoice/training/builder.py"
    echo "   🔧 Patched OmniVoice builder: flex_attention → eager"
fi

export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

# ---------------------------------------------------------------
# Step 1: Merge per-language JSONLs and split into train/dev
# ---------------------------------------------------------------
echo ""
echo "📋 Step 1: Merging JSONL manifests..."

mkdir -p "${DATA_DIR}"
MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
TRAIN_JSONL="${DATA_DIR}/train.jsonl"
DEV_JSONL="${DATA_DIR}/dev.jsonl"

# Merge all language JSONLs
cat /dev/null > "${MERGED_JSONL}"
for lang in fr ar zh; do
    SRC="${DEV_SYNTH_DIR}/train_${lang}.jsonl"
    if [ -f "${SRC}" ]; then
        echo "  Adding ${SRC} ($(wc -l < ${SRC}) samples)"
        cat "${SRC}" >> "${MERGED_JSONL}"
    else
        echo "  ⚠ Missing ${SRC}"
    fi
done

TOTAL=$(wc -l < "${MERGED_JSONL}")
echo "  Total merged samples: ${TOTAL}"

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
        python -m omnivoice.scripts.extract_audio_tokens \
        --input_jsonl "${SPLIT_JSONL}" \
        --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
        --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
        --tokenizer_path "${TOKENIZER_PATH}" \
        --nj_per_gpu 3 \
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
    "language_ratio": 0.8,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,

    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",

    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 500,
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.05,
    "warmup_steps": 0,

    "batch_tokens": 4096,
    "gradient_accumulation_steps": 16,
    "num_workers": 4,

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
    -m omnivoice.cli.train \
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
