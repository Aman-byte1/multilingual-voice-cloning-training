#!/bin/bash
# ============================================================
# PARALLEL Blind Test — Single-Model Runner (3 GPUs)
# Each GPU runs ONE model at a time via evaluate_blind_single.py
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
export COQUI_TOS_AGREED=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo "  PARALLEL BLIND TEST — 4 Models × 3 Languages × 4 Voices"
echo "  GPUs: 0, 1, 2"
echo "============================================================"

mkdir -p logs

# GPU 0: Chatterbox (all 3 languages)
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_blind_single.py \
    --model chatterbox --lang all \
    > logs/blind_chatterbox.log 2>&1 &
PID0=$!
echo "🚀 GPU 0: chatterbox (PID=$PID0)"

# GPU 1: VoxCPM (all 3 languages)
CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate_blind_single.py \
    --model voxcpm --lang all \
    > logs/blind_voxcpm.log 2>&1 &
PID1=$!
echo "🚀 GPU 1: voxcpm (PID=$PID1)"

# GPU 2: XTTS (all 3 languages)
CUDA_VISIBLE_DEVICES=2 python evaluation/evaluate_blind_single.py \
    --model xtts --lang all \
    > logs/blind_xtts.log 2>&1 &
PID2=$!
echo "🚀 GPU 2: xtts (PID=$PID2)"

echo ""
echo "  Monitor with:"
echo "    tail -f logs/blind_chatterbox.log"
echo "    tail -f logs/blind_voxcpm.log"
echo "    tail -f logs/blind_xtts.log"
echo ""

# Wait for all three
wait $PID0 && echo "✅ Chatterbox done" || echo "❌ Chatterbox failed"
wait $PID1 && echo "✅ VoxCPM done" || echo "❌ VoxCPM failed"
wait $PID2 && echo "✅ XTTS done" || echo "❌ XTTS failed"

# Phase 2: OmniVoice on GPU 0 (after Chatterbox finishes)
echo ""
echo "🚀 Phase 2: Running OmniVoice on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_blind_single.py \
    --model omnivoice --lang all \
    > logs/blind_omnivoice.log 2>&1 &
PID3=$!

# Qwen3 on GPU 1 (after VoxCPM finishes, fr+zh only)
echo "🚀 Phase 2: Running Qwen3 on GPU 1..."
CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate_blind_single.py \
    --model qwen3 --lang all \
    > logs/blind_qwen3.log 2>&1 &
PID4=$!

wait $PID3 && echo "✅ OmniVoice done" || echo "❌ OmniVoice failed"
wait $PID4 && echo "✅ Qwen3 done" || echo "❌ Qwen3 failed"

# Final comparison
echo ""
echo "============================================================"
echo "  📊 MERGING RESULTS"
echo "============================================================"

python3 - << 'PYEOF'
import json, os, glob

root = "./eval_blind_test"
summaries = glob.glob(os.path.join(root, "*/*/summary.json"))

if not summaries:
    print("No results found!")
    exit()

print(f"\n  {'Model':<14} {'Lang':<5} {'WER↓':>8} {'CER↓':>8} {'SIM↑':>8} {'Time(s)↓':>10}")
print("  " + "-" * 55)

for path in sorted(summaries):
    with open(path) as f:
        d = json.load(f)
    print(f"  {d['model']:<14} {d['lang']:<5} {d['WER']:>8.4f} {d['CER']:>8.4f} {d['Similarity']:>8.4f} {d['InferenceS']:>10.3f}")

print("  " + "=" * 55)
PYEOF
