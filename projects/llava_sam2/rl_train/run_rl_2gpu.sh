#!/bin/bash
#
# 2-GPU Distributed RL Training Script - Clean Test
#
# This script uses torchrun for distributed training across 2 GPUs
# with minimal configuration to validate the distributed setup
#

set -e

echo "========================================================================"
echo "Sa2VA Dual-Loop RL Training - 2 GPU Test (Clean Distributed Setup)"
echo "========================================================================"
echo ""

# Model and data paths
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
DATA_DIR="/data/xiaoyicheng/Sa2VA/data/GAR"
OUTPUT_DIR="./work_dirs/sa2va_rl_2gpu_test"
LOG_FILE="/tmp/sa2va_rl_2gpu.log"

# MINIMAL PARAMETERS
PER_GPU_BATCH_SIZE=1   # Batch size per GPU
NUM_GENERATIONS=2       # Minimum for GRPO
GRADIENT_ACCUM_STEPS=2  # Accumulate gradients
MAX_STEPS=3             # Quick test

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((2 * PER_GPU_BATCH_SIZE * GRADIENT_ACCUM_STEPS))

echo "2-GPU Configuration:"
echo "  GPUs: 0,1"
echo "  Per-GPU batch size: ${PER_GPU_BATCH_SIZE}"
echo "  Gradient accumulation steps: ${GRADIENT_ACCUM_STEPS}"
echo "  Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Num generations: ${NUM_GENERATIONS}"
echo "  Max steps: ${MAX_STEPS}"
echo ""

# Use only GPUs 0-1
export CUDA_VISIBLE_DEVICES=0,1

# Clear previous output
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
> ${LOG_FILE}

echo "Starting 2-GPU distributed training..."
echo "Log: ${LOG_FILE}"
echo ""

# Use torchrun - let it handle everything
torchrun \
    --nproc_per_node=2 \
    --master_port=29700 \
    projects/llava_sam2/rl_train/train_sa2va_dual_loop.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --num_generations ${NUM_GENERATIONS} \
    --learning_rate 1e-5 \
    --max_steps ${MAX_STEPS} \
    2>&1 | tee ${LOG_FILE}

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "✓ 2-GPU training completed successfully!"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "========================================================================"
echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | grep -E "^[0-1],"

echo ""
echo "Full log: ${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

exit ${EXIT_CODE}
