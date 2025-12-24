# Sa2VA RL Training - Progress Summary

## ✅ Completed Tasks

### 1. Dataset Loading (NVIDIA describe-anything)
**Status:** ✅ **COMPLETE**

- Successfully loaded NVIDIA describe-anything dataset from local Arrow files
- Implemented RLE mask decoding using pycocotools
- Implemented caption extraction from conversation format
- Created `dataset_nvidia.py` with `DescribeAnythingDataset` class

**Files Created:**
- `/data/xiaoyicheng/Sa2VA/projects/llava_sam2/rl_train/dataset_nvidia.py`
- `/data/xiaoyicheng/Sa2VA/projects/llava_sam2/rl_train/test_nvidia_quick.py` (test script)

**Dataset Statistics:**
- Location: `/data/xiaoyicheng/Sa2VA/data/GAR/`
- Part1: 3,108 samples from 3 arrow files
- Part2: Additional samples available
- Total across both parts: ~4,000+ samples

**Test Results:**
```
✓ Dataset loaded: 3108 samples
✓ Image loading works
✓ RLE mask decoding works
✓ Caption extraction works
✓ DataLoader batching works
✓ Ready for RL training!
```

### 2. Reward Functions (Basic)
**Status:** ✅ **COMPLETE**

- IOU reward for mask similarity
- METEOR reward for caption similarity

**Files:**
- `/data/xiaoyicheng/Sa2VA/projects/llava_sam2/rl_train/reward_functions.py`

### 3. EMA Model
**Status:** ✅ **COMPLETE**

- EMA wrapper with decay=0.999
- Proper parameter update logic

**Files:**
- `/data/xiaoyicheng/Sa2VA/projects/llava_sam2/rl_train/ema_model.py`

## 🔄 In Progress Tasks

### 4. LLM Judge Reward Function
**Status:** 🔄 **TODO**

**Requirements:**
- Implement caption similarity evaluation following describe-anything pattern
- Reference: `/data/xiaoyicheng/repos/describe-anything/evaluation/eval_model_outputs.py`
- Formula: `reward = 0.25 * METEOR + 0.75 * LLM_judge_score`
- Need to integrate OpenAI API or local LLM

**Implementation Plan:**
```python
def compute_llm_judge_score(pred_caption, gt_caption, llm_client):
    """
    Use LLM to evaluate caption similarity.
    Following describe-anything evaluation pattern:
    1. Generate questions based on gt_caption
    2. Use pred_caption to answer questions
    3. Score answers to compute similarity
    """
    # Implementation based on describe-anything eval
    pass

def compute_caption_reward_with_judge(pred_caption, gt_caption, llm_client):
    """
    Combined reward: 0.25 * METEOR + 0.75 * LLM_judge
    """
    meteor = compute_meteor(pred_caption, gt_caption)
    llm_score = compute_llm_judge_score(pred_caption, gt_caption, llm_client)
    return 0.25 * meteor + 0.75 * llm_score
```

### 5. R1-V Framework Integration
**Status:** 🔄 **TODO**

**Challenges:**
- R1-V is designed for Qwen2VL, not Sa2VA-4B
- Need to create custom `Sa2VAGRPOTrainer` based on `Qwen2VLGRPOTrainer`
- R1-V doesn't natively support dual-loop training
- Need to implement dual-loop logic in trainer

**R1-V Location:**
- Framework: `/data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src/open_r1/`
- GRPO implementation: `grpo.py`
- Trainer: `trainer/`

**Approach:**
1. Study `Qwen2VLGRPOTrainer` implementation
2. Create `Sa2VAGRPOTrainer` with:
   - Custom forward pass for Sa2VA model
   - Dual model management (policy + EMA)
   - Dual loop training logic
   - Custom reward computation

### 6. Complete Training Pipeline
**Status:** 🔄 **TODO**

**Components Needed:**
1. Sa2VAGRPOTrainer
2. LLM judge reward
3. Training script
4. Configuration files

## 📝 Implementation Order

### Next Steps (Priority Order):

1. **Implement LLM Judge Reward** (1-2 days)
   - Study describe-anything evaluation code
   - Implement question generation
   - Implement answer scoring
   - Test with sample captions

2. **Create Sa2VAGRPOTrainer** (2-3 days)
   - Study R1-V Qwen2VLGRPOTrainer
   - Adapt for Sa2VA model architecture
   - Implement EMA model integration
   - Implement dual-loop training logic
   - Add reward computation (IOU + caption)

3. **Write Training Script** (1 day)
   - Load pretrained Sa2VA-4B model
   - Setup LoRA for LLM
   - Configure trainable parameters (projector + LLM LoRA + SAM2 decoder)
   - Initialize EMA model
   - Setup dataloader
   - Training loop with logging

4. **Test Small-Scale Training** (1 day)
   - Test with 10-100 samples
   - Verify rewards are computed correctly
   - Verify EMA updates correctly
   - Verify model gradients flow correctly

5. **Full-Scale Training** (ongoing)
   - Train on complete dataset
   - Monitor rewards and losses
   - Evaluate checkpoints

## 🔧 Technical Details

### Model Architecture
- **Base Model:** Sa2VA-4B
- **Pretrained Weights:** `/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new`
- **Trainable Parts:**
  - Projector (mlp1)
  - LLM (LoRA adapter)
  - SAM2 decoder
- **Frozen Parts:**
  - Vision encoder
  - LLM base parameters
  - SAM2 encoder

### Training Details
- **Algorithm:** GRPO (Group Relative Policy Optimization)
- **Framework:** R1-V (TRL library)
- **Environment:** vlm conda environment
- **Models:** 2 models in EMA relationship (policy + EMA with decay=0.999)

### Dual Training Loops
1. **Loop 1 - Mask to Caption to Mask:**
   - Input: mask + image
   - Model 1 (policy): Generate caption from mask
   - Model 2 (EMA): Generate mask' from caption
   - Reward: IOU(mask, mask')

2. **Loop 2 - Caption to Mask to Caption:**
   - Input: caption + image
   - Model 1 (policy): Generate mask from caption
   - Model 2 (EMA): Generate caption' from mask
   - Reward: 0.25 * METEOR(caption, caption') + 0.75 * LLM_judge(caption, caption')

## 📚 Reference Code Locations

### Sa2VA Project
- Model: `/data/xiaoyicheng/Sa2VA/projects/llava_sam2/hf/models/modeling_sa2va_chat.py`
- Logprobs function: `_get_token_logprobs` (line 584-623)

### R1-V Framework
- GRPO: `/data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src/open_r1/grpo.py`
- Trainer: `/data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src/open_r1/trainer/`
- Config: Uses TRL's `GRPOConfig`, `GRPOTrainer`, `ModelConfig`

### describe-anything
- LLM Judge: `/data/xiaoyicheng/repos/describe-anything/evaluation/eval_model_outputs.py`

## 🎯 Current Priority

**Implement LLM Judge Reward Function**

This is the next critical component needed before we can integrate with R1-V and start training. Once the LLM judge is implemented, we can proceed with creating the Sa2VAGRPOTrainer and complete the training pipeline.

## 📊 Dataset Format (NVIDIA describe-anything)

```python
# Each sample contains:
{
    'image': PIL.Image (RGB),
    'mask': numpy.ndarray (H, W, bool),  # Decoded from RLE
    'caption': str,  # Extracted from conversations
    'category': list[str],  # Category labels
    'image_id': str
}

# Conversations format (raw):
[
    {"from": "human", "value": "<image>\nDescribe the masked region..."},
    {"from": "gpt", "value": "The bark is predominantly white..."}
]

# Mask RLE format (raw):
{
    'counts': 'kQ4121O0c5h0^J[O3;i3m1G6J6...',  # COCO compressed RLE
    'size': [H, W]  # [height, width]
}
```

## ✅ Environment Setup

- ✅ Conda environment: `vlm`
- ✅ Required libraries: pycocotools, pyarrow, datasets, PIL, numpy
- ✅ LD_LIBRARY_PATH configured for PIL compatibility
- ✅ HF mirror: https://hf-mirror.com (if needed)

## 🚀 Ready to Proceed

The dataloader is fully functional and tested. We can now proceed with:
1. Implementing the LLM judge reward
2. Integrating with R1-V framework
3. Starting RL training

All foundational components are in place and working correctly.
