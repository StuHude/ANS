#!/bin/bash
#
# Minimal parameter RL training script to avoid OOM
#
# Strategy:
# - Use single GPU (GPU 0)
# - Batch size = 1 (minimal)
# - Num generations = 2 (minimum for GRPO)
# - Short completion length
# - Gradient accumulation to simulate larger batch
#

set -e

echo "========================================================================"
echo "Sa2VA RL Training - Minimal Parameters (Anti-OOM)"
echo "========================================================================"
echo ""

# Model and data paths
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
DATA_DIR="/data/xiaoyicheng/Sa2VA/data/GAR"
OUTPUT_DIR="./work_dirs/sa2va_rl_minimal_test"
LOG_FILE="/tmp/sa2va_rl_minimal_run.log"

# MINIMAL PARAMETERS TO AVOID OOM
BATCH_SIZE=1           # Absolute minimum
NUM_GENERATIONS=2      # Minimum for GRPO (G=2)
NUM_EPOCHS=1
LEARNING_RATE=1e-5
MAX_STEPS=50           # Just test for 50 steps

# Use only GPU 0
export CUDA_VISIBLE_DEVICES=0

echo "Configuration (MINIMAL - Anti-OOM):"
echo "  GPU: 0 (NVIDIA RTX A6000, ~49GB)"
echo "  Model: Sa2VA-4B"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Num generations: ${NUM_GENERATIONS}"
echo "  Max steps: ${MAX_STEPS}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# Clear previous output
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
> ${LOG_FILE}

echo "Starting training..."
echo "Log: ${LOG_FILE}"
echo ""

# Activate conda environment and run
source /home/xiaoyicheng/miniconda3/etc/profile.d/conda.sh
conda activate vlm

python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --num_generations ${NUM_GENERATIONS} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --max_steps ${MAX_STEPS} \
    --save_steps 1000 \
    --save_total_limit 1 \
    2>&1 | tee ${LOG_FILE}

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ Training completed successfully!"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "========================================================================"
echo ""

# Analyze results
echo "Analysis:"
echo ""

# Check for OOM
OOM_COUNT=$(grep -c "CUDA out of memory\|OutOfMemoryError" ${LOG_FILE} || true)
if [ ${OOM_COUNT} -gt 0 ]; then
    echo "  ⚠ OOM detected: ${OOM_COUNT} times"
    echo "    → Need to reduce batch_size or num_generations further"
else
    echo "  ✓ No OOM errors"
fi

# Check for NaN/inf
NAN_COUNT=$(grep -c "NaN detected\|Inf detected" ${LOG_FILE} || true)
if [ ${NAN_COUNT} -gt 0 ]; then
    echo "  ⚠ NaN/Inf detected: ${NAN_COUNT} times (but handled by processors)"
else
    echo "  ✓ No NaN/Inf issues"
fi

# Check for successful steps
STEP_COUNT=$(grep -c "\[Step" ${LOG_FILE} || true)
echo "  Training steps completed: ${STEP_COUNT}"

echo ""
echo "Full log: ${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

exit ${EXIT_CODE}
