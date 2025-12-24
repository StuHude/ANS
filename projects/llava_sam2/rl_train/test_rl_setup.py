"""
测试RL训练设置的各个组件
"""

import sys
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

print("=" * 60)
print("Testing RL Training Setup")
print("=" * 60)

# Test 1: Dataset Loading
print("\n[Test 1] Dataset Loading...")
try:
    from projects.llava_sam2.rl_train.dataset_gar import GraspAnyRegionDataset

    dataset = GraspAnyRegionDataset(
        local_data_dir="/data/xiaoyicheng/Sa2VA/data/GAR",
        parts_to_load=['Fine-Grained-Dataset-Part1']  # Load only Part1 for testing
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Test sample
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"✓ Image shape: {sample['image'].size}")
    print(f"✓ Mask shape: {sample['mask'].shape}")
    print(f"✓ Caption: {sample['caption'][:50]}...")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Data Preprocessing
print("\n[Test 2] Data Preprocessing...")
try:
    from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor

    preprocessor = Sa2VADataPreprocessor()
    preprocessed = preprocessor.prepare_for_model(
        image=sample['image'],
        mask=sample['mask'],
        caption=sample['caption'],
        task="mask_to_caption"
    )
    print(f"✓ Preprocessed keys: {list(preprocessed.keys())}")
    print(f"✓ pixel_values shape: {preprocessed['pixel_values'].shape}")
    print(f"✓ prompt_masks shape: {preprocessed['prompt_masks'].shape}")
    print(f"✓ prompt_text: {preprocessed['prompt_text'][:80]}...")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Tokenization
print("\n[Test 3] Tokenization...")
try:
    from transformers import AutoTokenizer
    from projects.llava_sam2.rl_train.tokenization import Sa2VATemplateAndTokenizer

    model_path = "/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    template_tokenizer = Sa2VATemplateAndTokenizer(tokenizer, max_length=8196)

    # Create conversation
    conversation = [{
        'input': preprocessed['prompt_text'],
        'output': sample['caption']
    }]
    preprocessed['conversation'] = conversation

    # Tokenize
    tokenized = template_tokenizer(preprocessed, apply_template=True, tokenize=True)
    print(f"✓ Tokenized keys: {list(tokenized.keys())}")
    print(f"✓ input_ids length: {len(tokenized['input_ids'])}")
    print(f"✓ labels length: {len(tokenized['labels'])}")
except Exception as e:
    print(f"✗ Tokenization failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Model Loading
print("\n[Test 4] Model Loading...")
try:
    from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel

    print("  Loading Sa2VA model (this may take a while)...")
    model = Sa2VAChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"✓ Model loaded")
    print(f"✓ Vision model: {model.vision_model.__class__.__name__}")
    print(f"✓ Language model: {model.language_model.__class__.__name__}")
    print(f"✓ Grounding encoder: {model.grounding_encoder.__class__.__name__}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: LoRA Setup
print("\n[Test 5] LoRA Setup...")
try:
    # Apply LoRA
    model.wrap_llm_lora(r=128, lora_alpha=256, lora_dropout=0.05)
    print(f"✓ LoRA applied to LLM")

    # Check for LoRA modules
    lora_params = []
    for name, param in model.language_model.named_parameters():
        if 'lora' in name.lower():
            lora_params.append(name)
    print(f"✓ Found {len(lora_params)} LoRA parameters")
    if lora_params:
        print(f"  Example: {lora_params[0]}")
except Exception as e:
    print(f"✗ LoRA setup failed: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Parameter Freezing
print("\n[Test 6] Parameter Freezing...")
try:
    # Freeze vision encoder
    for param in model.vision_model.parameters():
        param.requires_grad = False

    # Freeze SAM2 encoder
    if hasattr(model.grounding_encoder, 'image_encoder'):
        for param in model.grounding_encoder.image_encoder.parameters():
            param.requires_grad = False

    # Keep mlp1 trainable
    for param in model.mlp1.parameters():
        param.requires_grad = True

    # Keep SAM2 decoder trainable
    if hasattr(model.grounding_encoder, 'mask_decoder'):
        for param in model.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True

    if hasattr(model.grounding_encoder, 'prompt_encoder'):
        for param in model.grounding_encoder.prompt_encoder.parameters():
            param.requires_grad = True

    # Keep text_hidden_fcs trainable
    for param in model.text_hidden_fcs.parameters():
        param.requires_grad = True

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Parameter freezing completed")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
except Exception as e:
    print(f"✗ Parameter freezing failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Reward Functions
print("\n[Test 7] Reward Functions...")
try:
    from projects.llava_sam2.rl_train.reward_functions import (
        compute_meteor, compute_iou, combined_caption_reward
    )

    # Test METEOR
    ref = "a person holding a red apple"
    hyp = "person with red apple"
    meteor_score = compute_meteor(ref, hyp)
    print(f"✓ METEOR score: {meteor_score:.4f}")

    # Test combined reward (without LLM judge)
    gt_captions = ["a person holding a red apple", "a blue car"]
    pred_captions = ["person with red apple", "blue vehicle"]
    rewards = combined_caption_reward(
        gt_captions=gt_captions,
        pred_captions=pred_captions,
        use_llm_judge=False
    )
    print(f"✓ Combined rewards (METEOR only): {[f'{r:.4f}' for r in rewards]}")
except Exception as e:
    print(f"✗ Reward functions failed: {e}")
    import traceback
    traceback.print_exc()

# Test 8: GRPO Trainer Imports
print("\n[Test 8] GRPO Trainer Components...")
try:
    from trl import GRPOConfig
    from trl.models import create_reference_model
    from projects.llava_sam2.rl_train.sa2va_grpo_trainer import Sa2VAGRPOTrainer

    print(f"✓ GRPOConfig imported")
    print(f"✓ create_reference_model imported")
    print(f"✓ Sa2VAGRPOTrainer imported")
except Exception as e:
    print(f"✗ GRPO trainer components failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All Tests Completed!")
print("=" * 60)
print("\nNext step: Run full RL training with:")
print("python projects/llava_sam2/rl_train/train_sa2va_rl.py \\")
print("    --model_path /data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new \\")
print("    --data_dir /data/xiaoyicheng/Sa2VA/data/GAR \\")
print("    --output_dir ./work_dirs/sa2va_rl_test \\")
print("    --batch_size 2 \\")
print("    --num_generations 2 \\")
print("    --num_epochs 1")
print("=" * 60)
