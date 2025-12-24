#!/usr/bin/env python3
"""
Debug script to check if LoRA parameters have requires_grad=True
and are properly registered in the optimizer.
"""
import sys
sys.path.insert(0, '/data/xiaoyicheng/Sa2VA')

import torch
from transformers import AutoTokenizer
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel


def load_sa2va_model(model_path, device='cuda'):
    """Load Sa2VA model"""
    print(f"Loading Sa2VA model from {model_path}...")
    model = Sa2VAChatModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    return model


def setup_lora(model, r=128, lora_alpha=256, lora_dropout=0.05):
    """Apply LoRA to LLM"""
    print(f"Setting up LoRA (r={r}, alpha={lora_alpha}, dropout={lora_dropout})...")
    model.wrap_llm_lora(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    print("✓ LoRA applied")
    return model


def freeze_parameters(model):
    """Freeze vision encoder and SAM2 encoder"""
    print("Freezing parameters...")

    # Freeze vision encoder
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print("  ✓ Vision encoder frozen")

    # Freeze SAM2 encoder
    if hasattr(model.grounding_encoder, 'image_encoder'):
        for param in model.grounding_encoder.image_encoder.parameters():
            param.requires_grad = False
        print("  ✓ SAM2 encoder frozen")

    # Keep mlp1 trainable
    for param in model.mlp1.parameters():
        param.requires_grad = True
    print("  ✓ Projector (mlp1) trainable")

    # LLM: LoRA adapters are trainable (handled by PEFT)
    print("  ✓ LLM LoRA adapters trainable")

    # Keep SAM2 decoder trainable
    if hasattr(model.grounding_encoder, 'mask_decoder'):
        for param in model.grounding_encoder.mask_decoder.parameters():
            param.requires_grad = True
        print("  ✓ SAM2 mask_decoder trainable")

    if hasattr(model.grounding_encoder, 'prompt_encoder'):
        for param in model.grounding_encoder.prompt_encoder.parameters():
            param.requires_grad = True
        print("  ✓ SAM2 prompt_encoder trainable")

    # Keep text_hidden_fcs trainable
    for param in model.text_hidden_fcs.parameters():
        param.requires_grad = True
    print("  ✓ text_hidden_fcs trainable")


def main():
    device = 'cuda'
    model_path = '/data/xiaoyicheng/Sa2VA/work_dirs/eval/Sa2VA-4B-epoch1-hf_new'

    # Load model
    model = load_sa2va_model(model_path, device=device)

    # Setup LoRA
    model = setup_lora(model, r=128, lora_alpha=256, lora_dropout=0.05)

    # Freeze parameters
    freeze_parameters(model)

    print("\n" + "="*70)
    print("PARAMETER ANALYSIS")
    print("="*70)

    # Count total and trainable params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")

    # Check LoRA parameters specifically
    print("\n" + "-"*70)
    print("LoRA PARAMETERS CHECK")
    print("-"*70)

    lora_params = 0
    lora_param_names = []
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params += param.numel()
            lora_param_names.append((name, param.numel(), param.requires_grad))

    print(f"\nFound {len(lora_param_names)} LoRA parameters")
    print(f"Total LoRA params: {lora_params:,}")

    # Print first 10 LoRA parameters
    print("\nFirst 10 LoRA parameters:")
    for name, numel, req_grad in lora_param_names[:10]:
        status = "✓" if req_grad else "✗"
        print(f"  {status} {name}: {numel:,} params, requires_grad={req_grad}")

    # Check if any LoRA params have requires_grad=False
    frozen_lora = [name for name, _, req_grad in lora_param_names if not req_grad]
    if frozen_lora:
        print(f"\n⚠ WARNING: {len(frozen_lora)} LoRA parameters have requires_grad=False!")
        print("First 5 frozen LoRA params:")
        for name in frozen_lora[:5]:
            print(f"  ✗ {name}")
    else:
        print("\n✓ All LoRA parameters have requires_grad=True")

    # Check other trainable modules
    print("\n" + "-"*70)
    print("OTHER TRAINABLE MODULES")
    print("-"*70)

    module_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad and 'lora' not in name.lower():
            # Extract module name (first part before '.')
            parts = name.split('.')
            module = parts[0] if parts else 'unknown'

            if module not in module_stats:
                module_stats[module] = {'params': 0, 'count': 0}
            module_stats[module]['params'] += param.numel()
            module_stats[module]['count'] += 1

    print("\nTrainable modules (excluding LoRA):")
    for module, stats in sorted(module_stats.items()):
        print(f"  {module}: {stats['count']} layers, {stats['params']:,} params")

    # Test forward pass
    print("\n" + "-"*70)
    print("TESTING FORWARD PASS")
    print("-"*70)

    # Create dummy inputs
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.preparing_for_generation(tokenizer)

    # Simple forward test
    batch_size = 1
    seq_len = 10

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
    dummy_attention_mask = torch.ones_like(dummy_input_ids)
    dummy_pixel_values = torch.randn(batch_size, 3, 448, 448, device=device, dtype=torch.bfloat16)

    # Create dummy prompt_masks (list of tensors)
    dummy_prompt_masks = [torch.randint(0, 2, (2, 16, 16), device=device, dtype=torch.bool) for _ in range(batch_size)]
    dummy_vp_overall_mask = torch.tensor([[True, True]], device=device)

    print("\nRunning forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(
                data={
                    'input_ids': dummy_input_ids,
                    'attention_mask': dummy_attention_mask,
                    'pixel_values': dummy_pixel_values,
                    'prompt_masks': dummy_prompt_masks,
                    'vp_overall_mask': dummy_vp_overall_mask,
                },
                mode='loss'
            )
        print("✓ Forward pass succeeded")
        print(f"  Output keys: {output.keys() if isinstance(output, dict) else 'not a dict'}")

        if hasattr(output, 'logits'):
            logits = output.logits
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits dtype: {logits.dtype}")
            print(f"  Logits range: [{logits.min():.4f}, {logits.max():.4f}]")
            print(f"  Has NaN: {torch.isnan(logits).any()}")
            print(f"  Has Inf: {torch.isinf(logits).any()}")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("DIAGNOSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
