#!/bin/bash
# ============================================================
# Parallel Evaluation Launcher
# Splits 416 samples across N parallel GPU processes
# Each process loads its own Chatterbox model (~4GB VRAM)
# With 48GB VRAM → safe to run 8 parallel processes
# Usage: bash evaluation/run_parallel_eval.sh
# ============================================================
set -e

LANG_CODE="${1:-zh}"
NUM_WORKERS="${2:-4}"
TOTAL_SAMPLES=416
OUTPUT_BASE="./eval_results/${LANG_CODE}_base"
CACHE_DIR="./data_cache"

CHUNK_SIZE=$(( (TOTAL_SAMPLES + NUM_WORKERS - 1) / NUM_WORKERS ))

echo "============================================"
echo "  Parallel Evaluation: ${LANG_CODE}"
echo "  Workers: ${NUM_WORKERS} | Samples: ${TOTAL_SAMPLES}"
echo "  Chunk size: ~${CHUNK_SIZE} samples/worker"
echo "  Output: ${OUTPUT_BASE}"
echo "============================================"

# Launch workers
PIDS=()
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    START=$((i * CHUNK_SIZE))
    END=$(( (i + 1) * CHUNK_SIZE ))
    if [ $END -gt $TOTAL_SAMPLES ]; then
        END=$TOTAL_SAMPLES
    fi

    echo ""
    echo "🚀 Launching worker $i: samples [$START:$END]"

    python evaluation/eval.py \
        --dataset ymoslem/acl-6060 \
        --split eval \
        --whisper-lang "$LANG_CODE" \
        --skip-lora \
        --cfg-weight 0.0 \
        --whisper-model large-v3 \
        --output-dir "$OUTPUT_BASE" \
        --cache-dir "$CACHE_DIR" \
        --resume \
        --start-idx "$START" \
        --end-idx "$END" \
        > "${OUTPUT_BASE}/worker_${i}.log" 2>&1 &

    PIDS+=($!)
    echo "   PID: ${PIDS[-1]}"
done

echo ""
echo "============================================"
echo "  All ${NUM_WORKERS} workers launched!"
echo "  Monitor with: tail -f ${OUTPUT_BASE}/worker_*.log"
echo "  Wait for all:  wait ${PIDS[*]}"
echo "============================================"

# Wait for all workers to finish
echo ""
echo "⏳ Waiting for all workers to complete..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "⚠ Worker PID $pid failed!"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo "⚠ $FAILED worker(s) failed. Check logs in ${OUTPUT_BASE}/worker_*.log"
else
    echo ""
    echo "============================================"
    echo "  ✅ All workers completed successfully!"
    echo "  Results: ${OUTPUT_BASE}/eval_summary.json"
    echo "============================================"
fi
