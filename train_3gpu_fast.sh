#!/bin/bash
# ============================================================
# FAST 3-GPU parallel training (~3-4 hours on 3x A40)
# ============================================================
# Fixes:
#   1. GPU assignment via accelerate --gpu_ids (not CUDA_VISIBLE_DEVICES)
#   2. Speed: batch_tokens 8192, grad_accum 4 → ~30s/step
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

LANGUAGES=("zh" "fr" "ar")
GPU_IDS=(0 1 2)
LORA_RANK=32
LORA_ALPHA=64
STEPS=400

# ──────────────────────────────────────────────────────────────
# Step 0: Kill any existing training processes
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Killing any existing training processes..."
echo "============================================================"
pkill -f "train_omnivoice_lora.py" 2>/dev/null || true
sleep 2
echo "  ✅ Clean"

# ──────────────────────────────────────────────────────────────
# Step 1: Apply patches
# ──────────────────────────────────────────────────────────────
python patch_omnivoice_attention.py --omnivoice-dir "${OMNIVOICE_DIR}"

# ──────────────────────────────────────────────────────────────
# Step 2: Verify tokens exist
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Verifying tokenized data..."
echo "============================================================"
for LANG in "${LANGUAGES[@]}"; do
    for split in train dev; do
        LST="./data/finetune_${LANG}/tokens/${split}/data.lst"
        if [ ! -f "${LST}" ] || [ ! -s "${LST}" ]; then
            echo "  ❌ Missing: ${LST}"
            echo "  Run clean_retokenize_and_train.sh first!"
            exit 1
        fi
        N=$(wc -l < "${LST}")
        echo "  ✅ ${LANG}/${split}: ${N} shards"
    done
done

# ──────────────────────────────────────────────────────────────
# Step 3: Write FAST training configs
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Writing fast training configs..."
echo "============================================================"
echo "  batch_tokens=8192, grad_accum=4, bf16, tf32"
echo "  Expected: ~30s/step × 400 steps ≈ 3.3 hours"

for LANG in "${LANGUAGES[@]}"; do
    CONFIG_DIR="./data/finetune_${LANG}/config"
    TOKEN_DIR="./data/finetune_${LANG}/tokens"
    mkdir -p "${CONFIG_DIR}"

    cat > "${CONFIG_DIR}/train_config.json" << 'HEREDOC'
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
    "batch_tokens": 8192,
    "gradient_accumulation_steps": 4,
    "num_workers": 4,
    "mixed_precision": "bf16",
    "allow_tf32": true,
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 100,
    "keep_last_n_checkpoints": 3
}
HEREDOC

    cat > "${CONFIG_DIR}/data_config.json" << EOF
{
    "train": [{"manifest_path": ["${TOKEN_DIR}/train/data.lst"]}],
    "dev": [{"manifest_path": ["${TOKEN_DIR}/dev/data.lst"]}]
}
EOF
done
echo "  ✅ Configs written"

# ──────────────────────────────────────────────────────────────
# Step 4: Launch all 3 — each on its own GPU
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Launching 3 parallel training jobs"
echo "  GPU 0→zh   GPU 1→fr   GPU 2→ar"
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

    echo "  [GPU ${GPU}] ${LANG}  →  ${LOG_FILE}"

    # Use accelerate --gpu_ids directly (NOT CUDA_VISIBLE_DEVICES)
    accelerate launch \
        --gpu_ids "${GPU}" \
        --num_processes 1 \
        train_omnivoice_lora.py \
        --lora_rank ${LORA_RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --train_config "${CONFIG_DIR}/train_config.json" \
        --data_config  "${CONFIG_DIR}/data_config.json" \
        --output_dir   "${OUTPUT_DIR}" \
        --vram_level   high \
        --use_rslora \
        --target_audio_modules \
    > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
    echo "    PID: ${PIDS[-1]}"
    # Small delay so accelerate configs don't collide
    sleep 3
done

echo ""
echo "  Verify GPU usage:  nvidia-smi"
echo "  Monitor training:  tail -f ./exp/train_zh.log ./exp/train_fr.log ./exp/train_ar.log"
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
