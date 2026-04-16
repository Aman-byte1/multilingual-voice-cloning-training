#!/bin/bash
# ============================================================
# OmniVoice Per-Language Fine-tuning 
# ============================================================
# Loops through an array of languages and fine-tunes OmniVoice 
# separately for each language. This approach isolates the LoRA
# weights per language, improving speaker similarity, WER, and CER
# within that specific language domain.

set -euo pipefail

# List of languages you want to fine-tune on
LANGUAGES=("zh" "fr" "ar")

# Common settings
GPU_IDS="0"
NUM_GPUS=1
MIN_SCORE="0.65"
LORA_RANK=32
LORA_ALPHA=64
STEPS=400 # slightly higher steps for better convergence on WER/CER
LEARNING_RATE="1e-4"

echo "============================================================"
echo "  Starting Per-Language Fine-Tuning Pipeline"
echo "  Target Languages: ${LANGUAGES[@]}"
echo "============================================================"

OMNIVOICE_DIR="./OmniVoice"
if [ ! -d "${OMNIVOICE_DIR}" ]; then
    echo "🚀 Initializing native OmniVoice architecture..."
    git clone https://github.com/k2-fsa/OmniVoice.git "${OMNIVOICE_DIR}"
    cd "${OMNIVOICE_DIR}"
    pip install -e .
    cd -
else
    echo "📦 OmniVoice repo already exists at ${OMNIVOICE_DIR}"
fi

# Fix: flex_attention crashes on GPUs with <128KB shared memory
python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

for LANG in "${LANGUAGES[@]}"; do
    echo ""
    echo "============================================================"
    echo "  [START] Fine-Tuning for Language: ${LANG^^}"
    echo "============================================================"

    # Set up language-specific paths
    DATA_DIR="./data/finetune_${LANG}"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    TOKEN_DIR="${DATA_DIR}/tokens"
    CONFIG_DIR="${DATA_DIR}/config"
    mkdir -p "${CONFIG_DIR}"

    # Step 1: Export strictly ${LANG} data
    MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
    TRAIN_JSONL="${DATA_DIR}/train.jsonl"
    DEV_JSONL="${DATA_DIR}/dev.jsonl"

    if [ ! -f "${MERGED_JSONL}" ]; then
        echo "--> Exporting ${LANG} filtered data from HF..."
        python download_dataset_from_hf.py \
            --split train \
            --min-score "${MIN_SCORE}" \
            --languages "${LANG}" \
            --output-dir "${DATA_DIR}/wavs" \
            --jsonl-path "${MERGED_JSONL}"
    else
        echo "--> Using existing manifest for ${LANG}: ${MERGED_JSONL}"
    fi

    # Split into train/dev
    python3 -c "
import random
with open('${MERGED_JSONL}') as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.90) # 90/10 split for slightly more training data (helps WER/CER)
with open('${TRAIN_JSONL}', 'w') as f:
    f.writelines(lines[:split])
with open('${DEV_JSONL}', 'w') as f:
    f.writelines(lines[split:])
print(f'Train size: {split} | Dev size: {len(lines) - split}')
"

    # Step 2: Tokenize Audio Data
    TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"
    for split_name in train dev; do
        SPLIT_JSONL="${TRAIN_JSONL}"
        [ "${split_name}" = "dev" ] && SPLIT_JSONL="${DEV_JSONL}"

        if [ -f "${TOKEN_DIR}/${split_name}/data.lst" ] && [ -s "${TOKEN_DIR}/${split_name}/data.lst" ]; then
            echo "--> Tokens already exist for ${split_name} (${LANG}). Skipping."
            continue
        fi

        echo "--> Tokenizing ${split_name} for ${LANG}..."
        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            python extract_audio_tokens_compat.py \
            --input_jsonl "${SPLIT_JSONL}" \
            --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 1 \
            --shuffle True
    done

    # Step 3: Config Setup
    # Tweak: "drop_cond_ratio": 0.05 (reduced from 0.1) forces the model 
    # to pay more attention to the prompt audio, significantly improving speaker similarity.
    cat > "${CONFIG_DIR}/train_config.json" << EOF
{
    "llm_name_or_path": "Qwen/Qwen3-0.6B",
    "audio_vocab_size": 1025,
    "audio_mask_id": 1024,
    "num_audio_codebook": 8,
    "audio_codebook_weights": [8, 8, 6, 6, 4, 4, 2, 2],
    "drop_cond_ratio": 0.05,
    "prompt_ratio_range": [0.3, 0.7],
    "mask_ratio_range": [0.0, 1.0],
    "language_ratio": 1.0,
    "use_pinyin_ratio": 0.0,
    "instruct_ratio": 0.0,
    "only_instruct_ratio": 0.0,
    "resume_from_checkpoint": null,
    "init_from_checkpoint": "k2-fsa/OmniVoice",
    "learning_rate": ${LEARNING_RATE},
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": ${STEPS},
    "seed": 42,
    "warmup_type": "ratio",
    "warmup_ratio": 0.05,
    "batch_tokens": 4096,
    "gradient_accumulation_steps": 16,
    "num_workers": 2,
    "mixed_precision": "no",
    "allow_tf32": false,
    "logging_steps": 10,
    "eval_steps": 50,
    "save_steps": 100,
    "keep_last_n_checkpoints": 3
}
EOF

    cat > "${CONFIG_DIR}/data_config.json" << EOF
{
    "train": [{"manifest_path": ["${TOKEN_DIR}/train/data.lst"]}],
    "dev": [{"manifest_path": ["${TOKEN_DIR}/dev/data.lst"]}]
}
EOF

    # Step 4: Execute Fine-Tuning
    export TORCHDYNAMO_DISABLE=1
    echo "--> Launching Accelerate for ${LANG}..."
    
    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        train_omnivoice_lora.py \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --train_config "${CONFIG_DIR}/train_config.json" \
        --data_config "${CONFIG_DIR}/data_config.json" \
        --output_dir "${OUTPUT_DIR}" \
        --use_rslora
        
    echo "============================================================"
    echo "  [DONE] Fine-Tuning for Language: ${LANG^^}"
    echo "  Checkpoints saved in: ${OUTPUT_DIR}"
    echo "============================================================"
done

echo "🎉 All languages fine-tuned successfully!"
