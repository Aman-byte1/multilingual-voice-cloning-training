#!/bin/bash
# ============================================================
# OmniVoice Per-Language Fine-tuning — PARALLEL on 3x A40
# ============================================================
# Each language gets its own GPU:
#   GPU 0 → zh  |  GPU 1 → fr  |  GPU 2 → ar
#
# Usage:
#   bash finetune_parallel_3gpu.sh
# ============================================================

set -euo pipefail

# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
LANGUAGES=("zh" "fr" "ar")
GPU_IDS=(0 1 2)

MIN_SCORE="0.65"
LORA_RANK=32
LORA_ALPHA=64
STEPS=400
LEARNING_RATE="1e-4"
VRAM_LEVEL="high"   # A40 has 48GB — use high preset

echo "============================================================"
echo "  PARALLEL Per-Language Fine-Tuning on 3x A40"
echo "  GPU 0 → zh  |  GPU 1 → fr  |  GPU 2 → ar"
echo "  Started: $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# Step 0: Ensure OmniVoice repo + patches
# ──────────────────────────────────────────────────────────────
OMNIVOICE_DIR="./OmniVoice"
if [ ! -d "${OMNIVOICE_DIR}" ]; then
    echo "🚀 Cloning OmniVoice..."
    git clone https://github.com/k2-fsa/OmniVoice.git "${OMNIVOICE_DIR}"
    cd "${OMNIVOICE_DIR}" && pip install -e . && cd -
else
    echo "📦 OmniVoice repo already at ${OMNIVOICE_DIR}"
fi

python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

# ──────────────────────────────────────────────────────────────
# Step 1: Prepare data for each language (sequential — light CPU work)
# ──────────────────────────────────────────────────────────────
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    DATA_DIR="./data/finetune_${LANG}"
    TOKEN_DIR="${DATA_DIR}/tokens"
    CONFIG_DIR="${DATA_DIR}/config"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    mkdir -p "${CONFIG_DIR}"

    MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
    TRAIN_JSONL="${DATA_DIR}/train.jsonl"
    DEV_JSONL="${DATA_DIR}/dev.jsonl"

    # Download if needed
    if [ ! -f "${MERGED_JSONL}" ]; then
        echo "--> Downloading ${LANG} data from HF..."
        python download_dataset_from_hf.py \
            --split train \
            --min-score "${MIN_SCORE}" \
            --languages "${LANG}" \
            --output-dir "${DATA_DIR}/wavs" \
            --jsonl-path "${MERGED_JSONL}"
    else
        echo "--> Using existing ${LANG} data: ${MERGED_JSONL}"
    fi

    # Train/dev split
    if [ ! -f "${TRAIN_JSONL}" ]; then
        python3 -c "
import random
with open('${MERGED_JSONL}') as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.90)
with open('${TRAIN_JSONL}', 'w') as f:
    f.writelines(lines[:split])
with open('${DEV_JSONL}', 'w') as f:
    f.writelines(lines[split:])
print(f'  ${LANG}: Train={split} Dev={len(lines)-split}')
"
    fi

    # Tokenize on its assigned GPU
    for split_name in train dev; do
        SPLIT_JSONL="${TRAIN_JSONL}"
        [ "${split_name}" = "dev" ] && SPLIT_JSONL="${DEV_JSONL}"

        if [ -f "${TOKEN_DIR}/${split_name}/data.lst" ] && [ -s "${TOKEN_DIR}/${split_name}/data.lst" ]; then
            echo "--> Tokens for ${LANG}/${split_name} exist. Skipping."
            continue
        fi

        echo "--> Tokenizing ${LANG}/${split_name} on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=${GPU} \
            python extract_audio_tokens_compat.py \
            --input_jsonl "${SPLIT_JSONL}" \
            --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 1 \
            --shuffle True
    done

    # Write configs
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
    "mixed_precision": "bf16",
    "allow_tf32": true,
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

    echo "  ✅ ${LANG} data prep complete"
done

# ──────────────────────────────────────────────────────────────
# Step 2: Launch all 3 training runs IN PARALLEL
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Launching 3 parallel training jobs..."
echo "============================================================"

PIDS=()

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    DATA_DIR="./data/finetune_${LANG}"
    CONFIG_DIR="${DATA_DIR}/config"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    LOG_FILE="./exp/train_${LANG}.log"
    mkdir -p "./exp"

    echo "  [GPU ${GPU}] ${LANG} → ${OUTPUT_DIR}  (log: ${LOG_FILE})"

    CUDA_VISIBLE_DEVICES=${GPU} \
    accelerate launch \
        --gpu_ids "0" \
        --num_processes 1 \
        train_omnivoice_lora.py \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --train_config "${CONFIG_DIR}/train_config.json" \
        --data_config "${CONFIG_DIR}/data_config.json" \
        --output_dir "${OUTPUT_DIR}" \
        --vram_level "${VRAM_LEVEL}" \
        --use_rslora \
        --target_audio_modules \
    > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "  All 3 jobs launched. Waiting for completion..."
echo "  Monitor logs with:  tail -f ./exp/train_zh.log ./exp/train_fr.log ./exp/train_ar.log"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 3: Wait for all jobs, report results
# ──────────────────────────────────────────────────────────────
FAILED=0
for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    PID="${PIDS[$i]}"

    if wait "${PID}"; then
        echo "  ✅ ${LANG} (GPU ${GPU_IDS[$i]}, PID ${PID}) — DONE"
    else
        echo "  ❌ ${LANG} (GPU ${GPU_IDS[$i]}, PID ${PID}) — FAILED (check ./exp/train_${LANG}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "${FAILED}" -eq 0 ]; then
    echo "  🎉 ALL 3 LANGUAGES FINE-TUNED SUCCESSFULLY — $(date)"
else
    echo "  ⚠  ${FAILED}/3 jobs failed — $(date)"
fi
echo "============================================================"
echo ""
echo "  Checkpoints:"
for LANG in "${LANGUAGES[@]}"; do
    echo "    ${LANG}: ./exp/omnivoice_finetuned_${LANG}/"
done
echo ""
