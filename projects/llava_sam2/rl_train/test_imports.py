#!/usr/bin/env python
"""Test imports to see where training script might be hanging"""

import sys
from pathlib import Path

print("1. Adding paths...")
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, '/data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src')
print("✓ Paths added")

print("\n2. Importing Sa2VA dataset components...")
from projects.llava_sam2.rl_train.dataset_gar import GraspAnyRegionDataset, collate_fn_sa2va_rl
print("✓ Dataset imported")

print("\n3. Importing preprocessor...")
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor
print("✓ Preprocessor imported")

print("\n4. Importing tokenization...")
from projects.llava_sam2.rl_train.tokenization import Sa2VATemplateAndTokenizer
print("✓ Tokenization imported")

print("\n5. Importing reward functions...")
from projects.llava_sam2.rl_train.reward_functions import (
    iou_reward_batch,
    combined_caption_reward,
    LLMJudge
)
print("✓ Reward functions imported")

print("\n6. Importing EMA model...")
from projects.llava_sam2.rl_train.ema_model import EMAModel
print("✓ EMA model imported")

print("\n7. Importing TRL...")
from trl import GRPOConfig, ModelConfig, get_peft_config
print("✓ TRL imported")

print("\n8. Importing transformers...")
from transformers import AutoTokenizer
print("✓ Transformers imported")

print("\n9. Importing Sa2VA model...")
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
print("✓ Sa2VA model imported")

print("\n10. All imports successful!")
print("="*60)
print("Training script should work. Issue might be in dataset loading.")
