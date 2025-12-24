# Sa2VA RL Training Implementation Summary

## ✅ COMPLETED Components

### 1. Dataset Loading & Preprocessing
- **File:** `dataset_gar.py` - GAR dataset loader from Arrow files ✅ TESTED
- **File:** `data_preprocessor.py` - Sa2VA format preprocessing ✅ TESTED
- **File:** `tokenization.py` - Template & tokenization (SFT-consistent) ✅ COMPLETE

### 2. Reward Functions
- **File:** `reward_functions.py` ✅ COMPLETE
  - IOU reward for mask evaluation
  - METEOR reward for caption evaluation
  - LLM judge (0.25×METEOR + 0.75×LLM_judge)
  - Loop 1: Always uses LLM judge (if available)
  - Loop 2: Controlled by `--use_llm_judge_loop2` flag (default: METEOR only)

### 3. Model Integration
- **File:** `train_sa2va_rl.py` ✅ COMPLETE
  - `load_sa2va_model()`: Loads Sa2VA-4B from HuggingFace checkpoint
  - `setup_lora()`: Applies LoRA to LLM component (r=128, alpha=256, same as SFT training)
  - `freeze_parameters()`: Freezes vision encoder and SAM2 encoder
  - Trainable: mlp1/projector, LLM LoRA, SAM2 decoder, text_hidden_fcs

### 4. GRPO Trainer (R1-V Based)
- **File:** `sa2va_grpo_trainer.py` ✅ COMPLETE
  - Adapted from R1-V's `Qwen2VLGRPOTrainer`
  - Integrated with Sa2VA's data preprocessing pipeline
  - Handles Sa2VA's input format (pixel_values, prompt_masks, etc.)
  - Supports reward function customization for dual-loop training

### 5. Training Pipeline
- **File:** `train_sa2va_rl.py` ✅ COMPLETE
  - Complete training loop with R1-V GRPO framework
  - Command-line interface with all parameters
  - Reference model handling (via create_reference_model or adapter disabling)
  - Checkpoint saving

## Dataset Path

**Current:** `/data/xiaoyicheng/Sa2VA/data/GAR`

**To change:** Edit `train_sa2va_rl.py` line 183 or use `--data_dir` argument

## Consistency with SFT Training

✅ **VERIFIED:** Tokenization and template are EXACTLY the same as SFT training
- Uses vicuna template from `pretrain_hf/templates.py`
- Uses `video_lisa_encode_fn` from `datasets/encode_fn.py`

## R1-V Framework

✅ **CONFIRMED:** Using R1-V framework (not implementing GRPO from scratch)
- Imports from `open_r1.trainer`
- Using `GRPOConfig` and `get_peft_config`

## Usage

### Basic Training (Loop 1: mask→caption)

```bash
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --data_dir /data/xiaoyicheng/Sa2VA/data/GAR \
    --output_dir ./work_dirs/sa2va_rl_training \
    --batch_size 4 \
    --num_generations 4 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --use_llm_judge
```

### Training with Loop 2 LLM Judge

To enable LLM judge for loop 2 (caption→mask→caption'):

```bash
python projects/llava_sam2/rl_train/train_sa2va_rl.py \
    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \
    --data_dir /data/xiaoyicheng/Sa2VA/data/GAR \
    --output_dir ./work_dirs/sa2va_rl_training \
    --batch_size 4 \
    --num_generations 4 \
    --learning_rate 1e-5 \
    --num_epochs 1 \
    --use_llm_judge \
    --use_llm_judge_loop2  # Enable LLM judge for loop 2
```

**Note:** By default, loop 2 uses 100% METEOR reward. Set `--use_llm_judge_loop2` to use combined reward (0.25×METEOR + 0.75×LLM_judge).

### Command-Line Arguments

- `--model_path`: Path to Sa2VA checkpoint (default: `/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new`)
- `--data_dir`: Path to GAR dataset (default: `/data/xiaoyicheng/Sa2VA/data/GAR`)
- `--output_dir`: Output directory for checkpoints (default: `./work_dirs/sa2va_rl_training`)
- `--batch_size`: Batch size per device (default: 4)
- `--num_generations`: Number of generations per prompt (G in GRPO paper) (default: 4)
- `--learning_rate`: Learning rate (default: 1e-5)
- `--num_epochs`: Number of training epochs (default: 1)
- `--ema_decay`: EMA decay rate (default: 0.999, not used in current R1-V implementation)
- `--use_llm_judge`: Use LLM judge for loop 1 caption rewards
- `--llm_judge_base_url`: Base URL for LLM judge API (default: `http://localhost:9100/v1`)
- `--use_llm_judge_loop2`: Use LLM judge for loop 2 (default: False, METEOR only)

## Testing

**Run dataset test:**
```bash
export PATH="/home/xiaoyicheng/miniconda3/bin:$PATH"
conda activate vlm
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python projects/llava_sam2/rl_train/test_gar_quick.py
python projects/llava_sam2/rl_train/test_preprocessor.py
```

## Implementation Notes

### Current Status
✅ **Loop 1 (mask→caption):** Fully implemented
- Uses mask as input, generates caption
- Reward: METEOR + LLM judge (if enabled)

⚠️ **Loop 2 (caption→mask):** Requires Sa2VA mask generation
- Would use caption as input, generate mask
- Reward: Controlled by `--use_llm_judge_loop2` flag
  - Default (False): 100% METEOR
  - True: 0.25×METEOR + 0.75×LLM_judge

### Dual-Loop Training

To implement dual-loop training:
1. Train with loop1_caption_reward for N epochs
2. Switch to loop2_caption_reward for N epochs
3. Alternate between loops

This can be done by modifying the reward_funcs in `train_sa2va_rl.py`:

```python
# Example: Alternate between loops
for epoch in range(total_epochs):
    if epoch % 2 == 0:
        trainer.reward_funcs = [reward_func_loop1]  # mask→caption
    else:
        trainer.reward_funcs = [reward_func_loop2]  # caption→mask
    trainer.train()
```

## Next Steps

1. ✅ Test basic training with small batch
2. ⚠️ Implement dual-loop training (requires Sa2VA mask generation support)
3. ⚠️ Run full training on complete dataset
4. ⚠️ Evaluate RL-trained model performance
