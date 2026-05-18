#!/bin/bash
# ============================================================
# FULL PIPELINE: Clone → Install → Download → Tokenize → Train
# OmniVoice per-language LoRA fine-tuning on 3x A40
# Dataset: amanuelbyte/omnivoice-best-of-n-dev-eval
# ============================================================
# Usage:
#   bash run_finetune_full.sh
# ============================================================
set -euo pipefail

echo "============================================================"
echo "  OmniVoice Full Fine-Tuning Pipeline"
echo "  Dataset: amanuelbyte/omnivoice-best-of-n-dev-eval"
echo "  GPUs: 3x A40  |  Languages: fr, ar, zh"
echo "  Started: $(date)"
echo "============================================================"

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
LANGUAGES=("fr" "ar" "zh")
GPU_IDS=(0 1 2)
HF_DATASET="amanuelbyte/omnivoice-best-of-n-dev-eval"
OMNIVOICE_DIR="./OmniVoice"
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

STEPS=400
LEARNING_RATE="1e-4"
LORA_RANK=32
LORA_ALPHA=64
MIN_SCORE="0.60"
BATCH_TOKENS=8192
GRAD_ACCUM=4

# ══════════════════════════════════════════════════════════════
# STEP 1: Clone OmniVoice
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 1/6: Clone OmniVoice ━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "${OMNIVOICE_DIR}" ]; then
    git clone https://github.com/k2-fsa/OmniVoice.git "${OMNIVOICE_DIR}"
    echo "  ✅ Cloned OmniVoice"
else
    echo "  ✓ OmniVoice already exists at ${OMNIVOICE_DIR}"
fi

# ══════════════════════════════════════════════════════════════
# STEP 2: Install dependencies
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 2/6: Install dependencies ━━━━━━━━━━━━━━━━━━━━━"

# Install OmniVoice in editable mode
cd "${OMNIVOICE_DIR}"
pip install -e . 2>&1 | tail -5
cd -

# Install additional training deps
pip install peft accelerate datasets soundfile jiwer faster-whisper speechbrain tqdm 2>&1 | tail -5

echo "  ✅ Dependencies installed"

# ══════════════════════════════════════════════════════════════
# STEP 3: Patch OmniVoice for eager attention (A40 compat)
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 3/6: Patch attention for A40 ━━━━━━━━━━━━━━━━━━"

python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"
export TORCHDYNAMO_DISABLE=1

echo "  ✅ Patches applied"

# ══════════════════════════════════════════════════════════════
# STEP 4: Download + split data per language
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 4/6: Download & prepare data ━━━━━━━━━━━━━━━━━━"

for LANG in "${LANGUAGES[@]}"; do
    DATA_DIR="./data/finetune_${LANG}"
    MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
    TRAIN_JSONL="${DATA_DIR}/train.jsonl"
    DEV_JSONL="${DATA_DIR}/dev.jsonl"
    mkdir -p "${DATA_DIR}"

    # Download
    if [ ! -f "${MERGED_JSONL}" ]; then
        echo "  → Downloading ${LANG} from ${HF_DATASET}..."
        python download_dataset_from_hf.py \
            --repo-id "${HF_DATASET}" \
            --split train \
            --min-score "${MIN_SCORE}" \
            --languages "${LANG}" \
            --output-dir "${DATA_DIR}/wavs" \
            --jsonl-path "${MERGED_JSONL}"
    else
        echo "  → ${LANG}: manifest exists ($(wc -l < "${MERGED_JSONL}") samples)"
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
print(f'  → ${LANG}: Train={split}  Dev={len(lines)-split}')
"
    fi
done

echo "  ✅ All data downloaded and split"

# ══════════════════════════════════════════════════════════════
# STEP 5: Tokenize audio → WebDataset shards
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 5/6: Tokenize audio ━━━━━━━━━━━━━━━━━━━━━━━━━━"

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    DATA_DIR="./data/finetune_${LANG}"
    TOKEN_DIR="${DATA_DIR}/tokens"
    TRAIN_JSONL="${DATA_DIR}/train.jsonl"
    DEV_JSONL="${DATA_DIR}/dev.jsonl"

    for split_name in train dev; do
        SPLIT_JSONL="${TRAIN_JSONL}"
        [ "${split_name}" = "dev" ] && SPLIT_JSONL="${DEV_JSONL}"

        if [ -f "${TOKEN_DIR}/${split_name}/data.lst" ] && [ -s "${TOKEN_DIR}/${split_name}/data.lst" ]; then
            echo "  → ${LANG}/${split_name}: tokens exist, skipping"
            continue
        fi

        echo "  → Tokenizing ${LANG}/${split_name} on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=${GPU} \
            python extract_audio_tokens_compat.py \
            --input_jsonl "${SPLIT_JSONL}" \
            --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 1 \
            --shuffle True
    done
done

echo "  ✅ Audio tokenized"

# ══════════════════════════════════════════════════════════════
# STEP 6: Train — 3 parallel LoRA jobs (1 per GPU)
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ Step 6/6: Launch training ━━━━━━━━━━━━━━━━━━━━━━━━━"

# Write configs
for LANG in "${LANGUAGES[@]}"; do
    DATA_DIR="./data/finetune_${LANG}"
    TOKEN_DIR="${DATA_DIR}/tokens"
    CONFIG_DIR="${DATA_DIR}/config"
    mkdir -p "${CONFIG_DIR}"

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
    "batch_tokens": ${BATCH_TOKENS},
    "gradient_accumulation_steps": ${GRAD_ACCUM},
    "num_workers": 4,
    "mixed_precision": "bf16",
    "allow_tf32": true,
    "logging_steps": 10,
    "eval_steps": 100,
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
done

# Launch parallel training
mkdir -p ./exp
PIDS=()

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    CONFIG_DIR="./data/finetune_${LANG}/config"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    LOG_FILE="./exp/train_${LANG}.log"
    mkdir -p "${OUTPUT_DIR}"

    echo "  [GPU ${GPU}] ${LANG^^} → ${OUTPUT_DIR}"

    accelerate launch \
        --gpu_ids "${GPU}" \
        --num_processes 1 \
        finetune_omnivoice_per_lang.py \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --train_config "${CONFIG_DIR}/train_config.json" \
        --data_config "${CONFIG_DIR}/data_config.json" \
        --output_dir "${OUTPUT_DIR}" \
        --vram_level high \
        --use_rslora \
        --target_audio_modules \
    > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    echo "          PID: ${PIDS[-1]}  |  Log: ${LOG_FILE}"
    sleep 3
done

echo ""
echo "  ┌──────────────────────────────────────────┐"
echo "  │  Monitor:                                │"
echo "  │    tail -f ./exp/train_fr.log            │"
echo "  │    tail -f ./exp/train_ar.log            │"
echo "  │    tail -f ./exp/train_zh.log            │"
echo "  │    watch -n2 nvidia-smi                  │"
echo "  └──────────────────────────────────────────┘"
echo ""

# Wait for all jobs
FAILED=0
for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    PID="${PIDS[$i]}"
    if wait "${PID}"; then
        echo "  ✅ ${LANG^^} — DONE"
    else
        echo "  ❌ ${LANG^^} — FAILED → tail -100 ./exp/train_${LANG}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "${FAILED}" -eq 0 ]; then
    echo "  🎉 ALL 3 LANGUAGES FINE-TUNED — $(date)"
    echo ""
    echo "  Checkpoints:"
    for LANG in "${LANGUAGES[@]}"; do
        echo "    ${LANG}: ./exp/omnivoice_finetuned_${LANG}/"
    done
    echo ""
    echo "  Evaluate:"
    echo "    python evaluate_lang.py --lang fr"
    echo "    python evaluate_lang.py --lang ar"
    echo "    python evaluate_lang.py --lang zh"
else
    echo "  ⚠  ${FAILED}/3 FAILED — $(date)"
fi
echo "============================================================"
