#!/bin/bash
# ============================================================
# Fix data.lst manifest format and (re)launch all 3 trainings
# ============================================================
# The extract_audio_tokens script writes data.lst with just paths,
# but OmniVoice training expects 4-column TSV:
#   tar_path \t jsonl_path \t num_items \t num_seconds
# ============================================================

set -euo pipefail

export TORCHDYNAMO_DISABLE=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

LORA_RANK=32
LORA_ALPHA=64
VRAM_LEVEL="high"

LANGUAGES=("zh" "fr" "ar")
GPU_IDS=(0 1 2)

# ──────────────────────────────────────────────────────────────
# Step 1: Fix all data.lst files
# ──────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Fixing data.lst manifest format for all languages"
echo "============================================================"

for LANG in "${LANGUAGES[@]}"; do
    TOKEN_DIR="./data/finetune_${LANG}/tokens"

    for split_name in train dev; do
        LST="${TOKEN_DIR}/${split_name}/data.lst"
        if [ ! -f "${LST}" ]; then
            echo "  ⚠ ${LANG}/${split_name}: data.lst not found — skipping"
            continue
        fi

        # Check if already in 4-column format (tab-separated)
        FIRST_LINE=$(head -1 "${LST}")
        NUM_TABS=$(echo "${FIRST_LINE}" | tr -cd '\t' | wc -c)
        if [ "${NUM_TABS}" -ge 3 ]; then
            echo "  ✅ ${LANG}/${split_name}: already in 4-column format"
            continue
        fi

        echo "  🔧 ${LANG}/${split_name}: converting to 4-column format..."

        python3 - "${LST}" <<'PYEOF'
import json
import os
import sys

lst_path = sys.argv[1]
with open(lst_path, "r") as f:
    lines = [l.strip() for l in f if l.strip()]

if not lines:
    print(f"    WARNING: {lst_path} is empty!", file=sys.stderr)
    sys.exit(0)

new_lines = []
for line in lines:
    # Current line might be just a path (jsonl or tar)
    parts = line.split("\t")
    if len(parts) >= 4:
        # Already correct format
        new_lines.append(line)
        continue

    # Single path — figure out if it's a jsonl or tar
    path = parts[0].strip()

    if path.endswith(".jsonl"):
        jsonl_path = path
        tar_path = path.replace("/txts/", "/audios/").replace(".jsonl", ".tar")
    elif path.endswith(".tar"):
        tar_path = path
        jsonl_path = path.replace("/audios/", "/txts/").replace(".tar", ".jsonl")
    else:
        print(f"    WARNING: unknown file type: {path}", file=sys.stderr)
        continue

    # Count items in the jsonl
    num_items = 0
    total_seconds = 0.0
    if os.path.exists(jsonl_path):
        with open(jsonl_path, "r") as jf:
            for jline in jf:
                jline = jline.strip()
                if not jline:
                    continue
                num_items += 1
                try:
                    obj = json.loads(jline)
                    # Try to get duration from various possible fields
                    dur = obj.get("duration", obj.get("num_seconds", obj.get("audio_duration", 0)))
                    total_seconds += float(dur)
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass

    # If we couldn't get duration from jsonl, estimate ~5s per sample
    if total_seconds < 0.01:
        total_seconds = num_items * 5.0

    new_lines.append(f"{tar_path}\t{jsonl_path}\t{num_items}\t{total_seconds:.2f}")

# Write back
with open(lst_path, "w") as f:
    for nl in new_lines:
        f.write(nl + "\n")

print(f"    → {len(new_lines)} entries, format: tar\\tjsonl\\titems\\tseconds")
PYEOF

    done
done

echo ""

# ──────────────────────────────────────────────────────────────
# Step 2: Verify all data.lst files
# ──────────────────────────────────────────────────────────────
echo "  Verifying manifests..."
ALL_OK=true
for LANG in "${LANGUAGES[@]}"; do
    for split_name in train dev; do
        LST="./data/finetune_${LANG}/tokens/${split_name}/data.lst"
        if [ ! -f "${LST}" ] || [ ! -s "${LST}" ]; then
            echo "  ❌ ${LANG}/${split_name}: missing or empty"
            ALL_OK=false
            continue
        fi
        N=$(wc -l < "${LST}")
        TABS=$(head -1 "${LST}" | tr -cd '\t' | wc -c)
        if [ "${TABS}" -lt 3 ]; then
            echo "  ❌ ${LANG}/${split_name}: still wrong format (${TABS} tabs)"
            ALL_OK=false
        else
            echo "  ✅ ${LANG}/${split_name}: ${N} shards (4-col format)"
        fi
    done
done

if [ "${ALL_OK}" != "true" ]; then
    echo ""
    echo "  ❌ Some manifests are invalid — cannot proceed."
    exit 1
fi

# ──────────────────────────────────────────────────────────────
# Step 3: Launch all 3 training runs in parallel
# ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  🚀 Launching all 3 training jobs in parallel"
echo "  GPU 0 → zh  |  GPU 1 → fr  |  GPU 2 → ar"
echo "============================================================"

PIDS=()
for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    CONFIG_DIR="./data/finetune_${LANG}/config"
    OUTPUT_DIR="./exp/omnivoice_finetuned_${LANG}"
    LOG_FILE="./exp/train_${LANG}.log"
    mkdir -p "${OUTPUT_DIR}"

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
echo "  Monitor:  tail -f ./exp/train_zh.log ./exp/train_fr.log ./exp/train_ar.log"
echo ""

# ──────────────────────────────────────────────────────────────
# Step 4: Wait and report
# ──────────────────────────────────────────────────────────────
FAILED=0
for i in "${!LANGUAGES[@]}"; do
    LANG="${LANGUAGES[$i]}"
    GPU="${GPU_IDS[$i]}"
    PID="${PIDS[$i]}"

    if wait "${PID}"; then
        echo "  ✅ ${LANG} (GPU ${GPU}, PID ${PID}) — DONE"
    else
        echo "  ❌ ${LANG} (GPU ${GPU}, PID ${PID}) — FAILED"
        echo "     Check: tail -50 ./exp/train_${LANG}.log"
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
