#!/bin/bash
#
# 4-GPU Distributed RL Training Script (GPU 0-3, same NUMA node)
#
# This script uses torchrun for distributed training across 4 GPUs
# All on the same NUMA node to avoid cross-NUMA communication issues
#

set -e

echo "========================================================================"
echo "Sa2VA Dual-Loop RL Training - 4 GPU Distributed (NUMA node 0)"
echo "========================================================================"
echo ""

# Model and data paths
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
DATA_DIR="/data/xiaoyicheng/Sa2VA/data/GAR"
OUTPUT_DIR="./work_dirs/sa2va_rl_4gpu_numa0"
LOG_FILE="/tmp/sa2va_rl_4gpu.log"

# MINIMAL PARAMETERS TO AVOID OOM
PER_GPU_BATCH_SIZE=1   # Batch size per GPU
NUM_GENERATIONS=2       # Minimum for GRPO
GRADIENT_ACCUM_STEPS=2  # Accumulate gradients
MAX_STEPS=5             # Quick test: just 5 steps

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((4 * PER_GPU_BATCH_SIZE * GRADIENT_ACCUM_STEPS))

echo "Multi-GPU Configuration:"
echo "  GPUs: 0,1,2,3 (4 GPUs on NUMA node 0)"
echo "  Per-GPU batch size: ${PER_GPU_BATCH_SIZE}"
echo "  Gradient accumulation steps: ${GRADIENT_ACCUM_STEPS}"
echo "  Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Num generations: ${NUM_GENERATIONS}"
echo "  Max steps: ${MAX_STEPS}"
echo ""

# Use only GPUs 0-3 (same NUMA node)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Clear previous output
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
> ${LOG_FILE}

echo "Starting distributed training..."
echo "Log: ${LOG_FILE}"
echo ""

# Use torchrun for distributed training
# --nproc_per_node: number of GPUs
# --master_port: port for inter-process communication (changed from 29500)
torchrun \
    --nproc_per_node=4 \
    --master_port=29600 \
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
    echo "✓ Distributed training completed successfully!"
else
    echo "✗ Training failed with exit code ${EXIT_CODE}"
fi
echo "========================================================================"
echo ""

# Check GPU usage
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | grep -E "^[0-3],"

echo ""
echo "Full log: ${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

exit ${EXIT_CODE}
