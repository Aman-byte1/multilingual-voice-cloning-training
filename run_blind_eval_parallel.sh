#!/bin/bash
# ============================================================
# PARALLEL Blind Test Evaluation — 3 GPUs, 5 Models
# Splits models across GPUs for maximum speed
# Expected time: ~1.5-2 hours (vs 6-8 sequential)
# ============================================================
set -euo pipefail

export TORCHDYNAMO_DISABLE=1
OMNIVOICE_DIR="./OmniVoice"
export PYTHONPATH="${OMNIVOICE_DIR}:${PYTHONPATH:-}"

echo "============================================================"
echo "  PARALLEL BLIND TEST — 5 Models × 3 Languages × 4 Voices"
echo "  GPUs: 0, 1, 2"
echo "============================================================"

# GPU 0: OmniVoice + XTTS (fastest models, ~1.5h combined)
CUDA_VISIBLE_DEVICES=0 python evaluation/evaluate_blind_all_models.py \
    --lang all --models omnivoice xtts \
    --output-dir ./eval_blind_test \
    > logs/blind_eval_gpu0.log 2>&1 &
PID0=$!
echo "🚀 GPU 0: omnivoice + xtts (PID=$PID0)"

# GPU 1: Chatterbox + VoxCPM (~2h combined)  
CUDA_VISIBLE_DEVICES=1 python evaluation/evaluate_blind_all_models.py \
    --lang all --models chatterbox voxcpm \
    --output-dir ./eval_blind_test \
    > logs/blind_eval_gpu1.log 2>&1 &
PID1=$!
echo "🚀 GPU 1: chatterbox + voxcpm (PID=$PID1)"

# GPU 2: Qwen3 (heaviest model, fr+zh only, ~1.5h)
CUDA_VISIBLE_DEVICES=2 python evaluation/evaluate_blind_all_models.py \
    --lang all --models qwen3 \
    --output-dir ./eval_blind_test \
    > logs/blind_eval_gpu2.log 2>&1 &
PID2=$!
echo "🚀 GPU 2: qwen3 (PID=$PID2)"

echo ""
echo "  All 3 jobs launched. Monitor with:"
echo "    tail -f logs/blind_eval_gpu0.log"
echo "    tail -f logs/blind_eval_gpu1.log"
echo "    tail -f logs/blind_eval_gpu2.log"
echo ""

# Wait for all to finish
mkdir -p logs
wait $PID0 && echo "✅ GPU 0 done" || echo "❌ GPU 0 failed"
wait $PID1 && echo "✅ GPU 1 done" || echo "❌ GPU 1 failed"  
wait $PID2 && echo "✅ GPU 2 done" || echo "❌ GPU 2 failed"

echo ""
echo "============================================================"
echo "  ALL DONE — Merging results..."
echo "============================================================"

# Merge all per-model results into one final table
python3 - << 'PYEOF'
import json, os, glob
from collections import defaultdict

root = "./eval_blind_test"
all_summaries = glob.glob(os.path.join(root, "*/*/summary.json"))

if not all_summaries:
    print("No results found!")
    exit()

print(f"\n{'Model':<14} {'Lang':<5} {'WER↓':>8} {'CER↓':>8} {'SIM↑':>8} {'Time(s)↓':>10}")
print("-" * 60)

for path in sorted(all_summaries):
    with open(path) as f:
        d = json.load(f)
    print(f"  {d['model']:<14} {d['lang']:<5} {d['WER']:>8.4f} {d['CER']:>8.4f} {d['Similarity']:>8.4f} {d['InferenceS']:>10.3f}")

print("=" * 60)
PYEOF
