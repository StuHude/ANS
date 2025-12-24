#!/bin/bash
#
# 8-GPU Distributed RL Training Script - FULL EPOCH
#
# This script uses torchrun for distributed training across 8 GPUs
# Training for a complete epoch on the expanded GAR dataset
#

set -e

echo "========================================================================"
echo "Sa2VA Dual-Loop RL Training - 8 GPU Full Epoch"
echo "========================================================================"
echo ""

# Model and data paths
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
DATA_DIR="/data/xiaoyicheng/Sa2VA/data/GAR"
OUTPUT_DIR="./work_dirs/sa2va_rl_8gpu_full_epoch"
LOG_FILE="/tmp/sa2va_rl_8gpu_full_epoch.log"

# TRAINING PARAMETERS
# With 8 GPUs, we can use larger total batch size
PER_GPU_BATCH_SIZE=1   # Batch size per GPU
NUM_GENERATIONS=2       # Minimum for GRPO
GRADIENT_ACCUM_STEPS=2  # Accumulate gradients to simulate larger batch

# Calculate effective batch size
EFFECTIVE_BATCH_SIZE=$((8 * PER_GPU_BATCH_SIZE * GRADIENT_ACCUM_STEPS))

echo "Multi-GPU Configuration:"
echo "  GPUs: 0,1,2,3,4,5,6,7 (8 GPUs)"
echo "  Per-GPU batch size: ${PER_GPU_BATCH_SIZE}"
echo "  Gradient accumulation steps: ${GRADIENT_ACCUM_STEPS}"
echo "  Effective batch size: ${EFFECTIVE_BATCH_SIZE}"
echo "  Num generations: ${NUM_GENERATIONS}"
echo "  Training mode: EPOCH-BASED (automatic steps calculation)"
echo ""

# Use all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL environment variables for better coordination and debugging
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800  # 30 minutes timeout
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# Clear previous output
rm -rf ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}
> ${LOG_FILE}

echo "Starting distributed training..."
echo "Log: ${LOG_FILE}"
echo ""

# Activate conda environment
source /home/xiaoyicheng/miniconda3/etc/profile.d/conda.sh
conda activate vlm

# Clean up any stale torchrun coordination files
rm -rf /tmp/torchelastic_*

# Use torchrun for distributed training
# --nproc_per_node: number of GPUs
# --master_port: port for inter-process communication (changed to avoid conflicts)
# --rdzv_timeout: timeout for rendezvous (in seconds)
torchrun \
    --nproc_per_node=8 \
    --master_port=29600 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:29600 \
    projects/llava_sam2/rl_train/train_sa2va_dual_loop.py \
    --model_path ${MODEL_PATH} \
    --data_dir ${DATA_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${PER_GPU_BATCH_SIZE} \
    --num_generations ${NUM_GENERATIONS} \
    --learning_rate 1e-5 \
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

# Analyze results
echo "Analysis:"
echo ""

# Check for OOM
OOM_COUNT=$(grep -c "CUDA out of memory\|OutOfMemoryError" ${LOG_FILE} || true)
if [ ${OOM_COUNT} -gt 0 ]; then
    echo "  ⚠ OOM detected: ${OOM_COUNT} times"
    echo "    → Need to reduce batch_size further or enable gradient checkpointing"
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

# Check training steps
STEP_COUNT=$(grep -c "{'loss':" ${LOG_FILE} || true)
echo "  Training steps completed: ${STEP_COUNT}"

# Check GPU usage
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv | grep -E "^[0-7],"

echo ""
echo "Full log: ${LOG_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

exit ${EXIT_CODE}
