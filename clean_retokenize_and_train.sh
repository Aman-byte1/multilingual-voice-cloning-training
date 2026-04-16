#!/bin/bash
# ============================================================
# Clean restart: wipe tokens → re-tokenize → train all 3 GPUs
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"
LORA_RANK=32
LORA_ALPHA=64
VRAM_LEVEL="high"

LANGUAGES=("zh" "fr" "ar")
GPU_IDS=(0 1 2)

# ──────────────────────────────────────────────────────────────
# Step 0: Apply OmniVoice attention patch
# ──────────────────────────────────────────────────────────────
python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"

# ──────────────────────────────────────────────────────────────
# Step 1: Wipe all stale token directories
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Wiping stale token directories..."
echo "============================================================"
for LANG in "${LANGUAGES[@]}"; do
    TOKEN_DIR="./data/finetune_${LANG}/tokens"
    if [ -d "${TOKEN_DIR}" ]; then
        echo "  Removing ${TOKEN_DIR}"
        rm -rf "${TOKEN_DIR}"
    fi
done
echo "  ✅ Clean"

# ──────────────────────────────────────────────────────────────
# Step 2: Verify JSONL splits exist for all languages
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying JSONL splits..."
echo "============================================================"
ALL_OK=true
for LANG in "${LANGUAGES[@]}"; do
    DATA_DIR="./data/finetune_${LANG}"
    MERGED="${DATA_DIR}/merged_all.jsonl"
    TRAIN="${DATA_DIR}/train.jsonl"
    DEV="${DATA_DIR}/dev.jsonl"

    # Re-create split if missing
    for F in "${TRAIN}" "${DEV}"; do
        if [ ! -f "${F}" ] || [ ! -s "${F}" ]; then
            echo "  ⚠ ${F} missing — re-splitting from merged..."
            python3 - <<PYEOF
import random, sys
merged = "${MERGED}"
train_p = "${DATA_DIR}/train.jsonl"
dev_p   = "${DATA_DIR}/dev.jsonl"
if not __import__('os').path.exists(merged):
    print(f"ERROR: {merged} not found!", file=sys.stderr)
    sys.exit(1)
with open(merged, encoding="utf-8") as f:
    lines = f.readlines()
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.90)
with open(train_p, "w", encoding="utf-8") as f:
    f.writelines(lines[:split])
with open(dev_p, "w", encoding="utf-8") as f:
    f.writelines(lines[split:])
print(f"  ${LANG}: Train={split} Dev={len(lines)-split}")
PYEOF
            break
        fi
    done

    N_TRAIN=$(wc -l < "${TRAIN}" 2>/dev/null || echo 0)
    N_DEV=$(wc -l < "${DEV}" 2>/dev/null || echo 0)
    echo "  ${LANG}: train=${N_TRAIN} dev=${N_DEV}"
    if [ "${N_TRAIN}" -eq 0 ] || [ "${N_DEV}" -eq 0 ]; then
        echo "  ❌ ${LANG} has empty splits!"
        ALL_OK=false
    fi
done

if [ "${ALL_OK}" != "true" ]; then
    echo "❌ Cannot proceed — fix JSONL splits first."
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Step 3: Re-tokenize all splits (sequential, one language at a time)
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Re-tokenizing all splits..."
echo "============================================================"

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    DATA_DIR="./data/finetune_${LANG}"
    TOKEN_DIR="${DATA_DIR}/tokens"

    for split_name in train dev; do
        SPLIT_JSONL="${DATA_DIR}/${split_name}.jsonl"
        LST="${TOKEN_DIR}/${split_name}/data.lst"

        echo ""
        echo "  --> Tokenizing ${LANG}/${split_name} on GPU ${GPU}..."
        CUDA_VISIBLE_DEVICES=${GPU} \
            python extract_audio_tokens_compat.py \
            --input_jsonl "${SPLIT_JSONL}" \
            --tar_output_pattern "${TOKEN_DIR}/${split_name}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split_name}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 1 \
            --shuffle True

        # Validate
        N=$(wc -l < "${LST}" 2>/dev/null || echo 0)
        if [ "${N}" -eq 0 ]; then
            echo "  ❌ Tokenization produced 0 shards for ${LANG}/${split_name}!"
            exit 1
        fi
        # Show first line to confirm format
        FIRST=$(head -1 "${LST}")
        echo "  ✅ ${LANG}/${split_name}: ${N} shards"
        echo "     Sample: ${FIRST:0:120}"
    done

    # Write config files
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
    "learning_rate": 1e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "steps": 400,
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

    echo "  ✅ ${LANG} ready"
done

# ──────────────────────────────────────────────────────────────
# Step 4: Launch all 3 training jobs in parallel
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Launching all 3 training jobs in parallel"
echo "  GPU 0→zh  GPU 1→fr  GPU 2→ar"
echo "============================================================"
mkdir -p ./exp
PIDS=()

for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    CONFIG_DIR="./data/finetune_${LANG}/config"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    LOG_FILE="./exp/train_${LANG}.log"
    mkdir -p "${OUTPUT_DIR}"

    echo "  [GPU ${GPU}] ${LANG}  log: ${LOG_FILE}"

    CUDA_VISIBLE_DEVICES=${GPU} \
    accelerate launch \
        --gpu_ids "0" \
        --num_processes 1 \
        train_omnivoice_lora.py \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --train_config "${CONFIG_DIR}/train_config.json" \
        --data_config  "${CONFIG_DIR}/data_config.json" \
        --output_dir   "${OUTPUT_DIR}" \
        --vram_level   "${VRAM_LEVEL}" \
        --use_rslora \
        --target_audio_modules \
    > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
done

echo ""
echo "  Monitor: tail -f ./exp/train_zh.log ./exp/train_fr.log ./exp/train_ar.log"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 5: Wait and report
# ──────────────────────────────────────────────────────────────
FAILED=0
for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    PID="${PIDS[$i]}"
    if wait "${PID}"; then
        echo "  ✅ ${LANG} — DONE"
    else
        echo "  ❌ ${LANG} — FAILED  →  tail -50 ./exp/train_${LANG}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================================"
if [ "${FAILED}" -eq 0 ]; then
    echo "  🎉 ALL 3 DONE — $(date)"
else
    echo "  ⚠  ${FAILED}/3 FAILED — $(date)"
fi
echo "============================================================"
