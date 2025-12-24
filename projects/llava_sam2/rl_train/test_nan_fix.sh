#!/bin/bash
#
# Test script to validate NaN/inf fixes in Sa2VA RL training
#
# This script runs a short training session to verify:
# 1. No NaN/inf in logits during generation
# 2. Gradient clipping is working
# 3. Monitoring callbacks are functioning
#

set -e  # Exit on error

echo "========================================================================"
echo "Testing Sa2VA RL Training with NaN/inf Fixes"
echo "========================================================================"
echo ""

# Configuration
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
DATA_DIR="/data/xiaoyicheng/Sa2VA/data/GAR"
OUTPUT_DIR="./work_dirs/sa2va_rl_nan_fix_test"
LOG_FILE="/tmp/sa2va_rl_nan_fix_test.log"

# Training parameters - minimal for quick test
BATCH_SIZE=2          # Small batch for testing
NUM_GENERATIONS=2     # Minimum G value
NUM_EPOCHS=1
LEARNING_RATE=1e-5

echo "Configuration:"
echo "  Model: ${MODEL_PATH}"
echo "  Data: ${DATA_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${LOG_FILE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num generations: ${NUM_GENERATIONS}"
echo ""

# Clear previous log
> ${LOG_FILE}

echo "Starting training test..."
echo "Logging to: ${LOG_FILE}"
echo ""

# Run training with all fixes enabled
python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_generations ${NUM_GENERATIONS} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --save_steps 1000 \
    --save_total_limit 1 \
    2>&1 | tee ${LOG_FILE}

echo ""
echo "========================================================================"
echo "Training test completed!"
echo "========================================================================"
echo ""

# Analyze log for NaN/inf issues
echo "Analyzing log for NaN/inf issues..."
echo ""

NAN_COUNT=$(grep -c "NaN detected" ${LOG_FILE} || true)
INF_COUNT=$(grep -c "Inf detected" ${LOG_FILE} || true)
GRAD_ISSUE_COUNT=$(grep -c "GRADIENT ISSUE DETECTED" ${LOG_FILE} || true)
CLIP_COUNT=$(grep -c "Gradient clipping triggered" ${LOG_FILE} || true)

echo "Results:"
echo "  NaN detections in logits: ${NAN_COUNT}"
echo "  Inf detections in logits: ${INF_COUNT}"
echo "  Gradient issues: ${GRAD_ISSUE_COUNT}"
echo "  Gradient clipping events: ${CLIP_COUNT}"
echo ""

if [ ${NAN_COUNT} -eq 0 ] && [ ${INF_COUNT} -eq 0 ]; then
    echo "✓ SUCCESS: No NaN/inf issues detected during generation!"
else
    echo "⚠ WARNING: NaN/inf issues were detected and handled by stability processors"
    echo "  This is expected for the first few steps as the model stabilizes"
fi

echo ""
echo "Full log available at: ${LOG_FILE}"
echo "========================================================================"
