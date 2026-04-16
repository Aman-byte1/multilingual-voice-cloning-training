#!/bin/bash
# ============================================================
# Recovery: Re-split → Re-tokenize → Re-train fr & ar
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"
LORA_RANK=32
LORA_ALPHA=64
STEPS=400
LEARNING_RATE="1e-4"
VRAM_LEVEL="high"

# Map: language → GPU
declare -A LANG_GPU=( ["fr"]=1 ["ar"]=2 )

for LANG in fr ar; do
    GPU="${LANG_GPU[$LANG]}"
    DATA_DIR="./data/finetune_${LANG}"
    TOKEN_DIR="${DATA_DIR}/tokens"
    CONFIG_DIR="${DATA_DIR}/config"
    MERGED_JSONL="${DATA_DIR}/merged_all.jsonl"
    TRAIN_JSONL="${DATA_DIR}/train.jsonl"
    DEV_JSONL="${DATA_DIR}/dev.jsonl"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"

    echo "============================================================"
    echo "  RECOVERY: ${LANG} on GPU ${GPU}"
    echo "============================================================"

    # ── Validate merged JSONL ──────────────────────────────────────
    if [ ! -f "${MERGED_JSONL}" ]; then
        echo "ERROR: ${MERGED_JSONL} missing — re-run download first."
        exit 1
    fi
    N_MERGED=$(wc -l < "${MERGED_JSONL}")
    echo "  merged_all.jsonl: ${N_MERGED} lines"
    if [ "${N_MERGED}" -eq 0 ]; then
        echo "ERROR: ${MERGED_JSONL} is empty!"
        exit 1
    fi

    # ── Re-split using python file (avoids heredoc quoting issues) ──
    echo "--> Splitting ${LANG}..."
    python3 - <<PYEOF
import random, sys
merged  = "${MERGED_JSONL}"
train_p = "${TRAIN_JSONL}"
dev_p   = "${DEV_JSONL}"
with open(merged, encoding="utf-8") as f:
    lines = f.readlines()
if not lines:
    print("ERROR: merged JSONL is empty!", file=sys.stderr)
    sys.exit(1)
random.seed(42)
random.shuffle(lines)
split = int(len(lines) * 0.90)
with open(train_p, "w", encoding="utf-8") as f:
    f.writelines(lines[:split])
with open(dev_p, "w", encoding="utf-8") as f:
    f.writelines(lines[split:])
print(f"  Train: {split} | Dev: {len(lines)-split}")
PYEOF

    N_TRAIN=$(wc -l < "${TRAIN_JSONL}")
    echo "  train.jsonl: ${N_TRAIN} lines"
    if [ "${N_TRAIN}" -eq 0 ]; then
        echo "ERROR: train split is still empty after re-split!"
        exit 1
    fi

    # ── Re-tokenize (wipe stale empty shards first) ─────────────────
    for split_name in train dev; do
        SPLIT_JSONL="${TRAIN_JSONL}"
        [ "${split_name}" = "dev" ] && SPLIT_JSONL="${DEV_JSONL}"

        # Remove any stale empty data.lst so the skip check won't fire
        LST="${TOKEN_DIR}/${split_name}/data.lst"
        if [ -f "${LST}" ] && [ ! -s "${LST}" ]; then
            echo "--> Removing empty ${LST}"
            rm -f "${LST}"
            rm -rf "${TOKEN_DIR}/${split_name}/audios" "${TOKEN_DIR}/${split_name}/txts"
        fi

        if [ -f "${LST}" ] && [ -s "${LST}" ]; then
            echo "--> Tokens for ${LANG}/${split_name} already exist. Skipping."
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

        N_SHARDS=$(wc -l < "${LST}" 2>/dev/null || echo 0)
        echo "  Shards written: ${N_SHARDS}"
        if [ "${N_SHARDS}" -eq 0 ]; then
            echo "ERROR: tokenization produced 0 shards for ${LANG}/${split_name}!"
            exit 1
        fi
    done

    echo "  ✅ ${LANG} data ready"
done

# ── Launch fr and ar in parallel ────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Re-launching fr (GPU 1) and ar (GPU 2) in parallel..."
echo "============================================================"

PIDS=()
for LANG in fr ar; do
    GPU="${LANG_GPU[$LANG]}"
    DATA_DIR="./data/finetune_${LANG}"
    CONFIG_DIR="${DATA_DIR}/config"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    LOG_FILE="./exp/train_${LANG}.log"

    echo "  [GPU ${GPU}] ${LANG} → log: ${LOG_FILE}"

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
echo "  Monitoring: tail -f ./exp/train_fr.log ./exp/train_ar.log"

FAILED=0
for i in 0 1; do
    LANGS=(fr ar)
    GPUS=(1 2)
    LANG="${LANGS[$i]}"
    if wait "${PIDS[$i]}"; then
        echo "  ✅ ${LANG} — DONE"
    else
        echo "  ❌ ${LANG} — FAILED (check ./exp/train_${LANG}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ "${FAILED}" -eq 0 ]; then
    echo "🎉 fr and ar fine-tuning complete!"
else
    echo "⚠  ${FAILED} job(s) failed."
fi
