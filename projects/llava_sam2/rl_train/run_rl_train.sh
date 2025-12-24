#!/bin/bash

# Sa2VA RL Training Script
# Usage: bash projects/llava_sam2/rl_train/run_rl_train.sh

set -e

# ============ Configuration ============
# Model
MODEL_PATH="/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"

# Data
# Option 1: Load from HuggingFace (online)
# DATASET_NAME="HaochenWang/Grasp-Any-Region-Dataset"
# CACHE_DIR="./data/grasp_any_region_cache"
# LOCAL_DATA_DIR=""

# Option 2: Load from local Arrow files (offline) - RECOMMENDED
# Set LOCAL_DATA_DIR to the directory containing Fine-Grained-Dataset-Part* folders
LOCAL_DATA_DIR="/path/to/your/grasp_dataset"  # CHANGE THIS to your actual path
DATASET_NAME="HaochenWang/Grasp-Any-Region-Dataset"  # Ignored when LOCAL_DATA_DIR is set
CACHE_DIR="./data/grasp_any_region_cache"

# Specify which parts to load (optional, if not set, loads all available parts)
# PARTS_TO_LOAD="Fine-Grained-Dataset-Part1 Fine-Grained-Dataset-Part2 Fine-Grained-Dataset-Part3 Fine-Grained-Dataset-Part4 Fine-Grained-Dataset-Part5 Fine-Grained-Dataset-Part6"
PARTS_TO_LOAD=""  # Leave empty to auto-detect and load all parts

# Output
OUTPUT_DIR="./outputs/sa2va_grpo_$(date +%Y%m%d_%H%M%S)"

# Training
NUM_EPOCHS=2
PER_DEVICE_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=2
LEARNING_RATE=1e-5
WARMUP_STEPS=100
MAX_GRAD_NORM=1.0

# GRPO
NUM_GENERATIONS=4
KL_COEF=0.1
CLIP_RANGE=0.2

# EMA
EMA_DECAY=0.999

# Task weights
MASK_TO_CAPTION_WEIGHT=1.0
CAPTION_TO_MASK_WEIGHT=1.0

# Logging
LOGGING_STEPS=10
SAVE_STEPS=500
WANDB_PROJECT="sa2va-rl"
WANDB_RUN_NAME="sa2va_grpo_$(date +%Y%m%d_%H%M%S)"

# Distributed
NUM_GPUS=8

# ============ Check requirements ============
echo "Checking Python environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Install dependencies if needed
echo "Installing dependencies..."
pip install -q datasets wandb nltk tqdm

# Download NLTK data
python -c "import nltk; nltk.download('wordnet', quiet=True); nltk.download('omw-1.4', quiet=True)" || true

echo "Setup complete!"
echo ""

# ============ Print configuration ============
echo "==========================================="
echo "Sa2VA RL Training Configuration"
echo "==========================================="
echo "Model Path: $MODEL_PATH"
echo "Dataset: $DATASET_NAME"
echo "Output Dir: $OUTPUT_DIR"
echo "Num GPUs: $NUM_GPUS"
echo "Num Epochs: $NUM_EPOCHS"
echo "Batch Size: $PER_DEVICE_BATCH_SIZE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Learning Rate: $LEARNING_RATE"
echo "Num Generations (G): $NUM_GENERATIONS"
echo "KL Coefficient: $KL_COEF"
echo "EMA Decay: $EMA_DECAY"
echo "==========================================="
echo ""

# ============ Build arguments ============
COMMON_ARGS="--model_path $MODEL_PATH \
    --dataset_name $DATASET_NAME \
    --cache_dir $CACHE_DIR \
    --output_dir $OUTPUT_DIR \
    --num_epochs $NUM_EPOCHS \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --warmup_steps $WARMUP_STEPS \
    --max_grad_norm $MAX_GRAD_NORM \
    --num_generations $NUM_GENERATIONS \
    --kl_coef $KL_COEF \
    --clip_range $CLIP_RANGE \
    --ema_decay $EMA_DECAY \
    --mask_to_caption_weight $MASK_TO_CAPTION_WEIGHT \
    --caption_to_mask_weight $CAPTION_TO_MASK_WEIGHT \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME"

# Add local data directory if specified
if [ -n "$LOCAL_DATA_DIR" ]; then
    COMMON_ARGS="$COMMON_ARGS --local_data_dir $LOCAL_DATA_DIR"
    echo "Using local dataset from: $LOCAL_DATA_DIR"
fi

# Add parts to load if specified
if [ -n "$PARTS_TO_LOAD" ]; then
    COMMON_ARGS="$COMMON_ARGS --parts_to_load $PARTS_TO_LOAD"
    echo "Loading specific parts: $PARTS_TO_LOAD"
fi

# ============ Run training ============
if [ $NUM_GPUS -eq 1 ]; then
    echo "Running single-GPU training..."
    python projects/llava_sam2/rl_train/train_sa2va_rl.py $COMMON_ARGS
else
    echo "Running multi-GPU training with $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=127.0.0.1 \
        --master_port=29500 \
        projects/llava_sam2/rl_train/train_sa2va_rl.py $COMMON_ARGS
fi

echo ""
echo "Training completed! Checkpoints saved to: $OUTPUT_DIR"
