"""
Sa2VA Single-Stage RL Training - Mask Captioning with LLM Reward

Task: mask + image -> caption
Reward: LLM-based evaluation (following describe-anything style)

This is a simpler RL training setup compared to dual-loop:
- Input: image + mask region
- Output: caption describing the masked region
- Reward: LLM evaluates caption similarity (via vLLM server)

Prerequisites:
1. Start vLLM server on a separate GPU:
   CUDA_VISIBLE_DEVICES=7 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
       --tensor-parallel-size 1 --port 9100 --max-model-len 4096
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import List, Optional
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.rl_train.dataset_gar import GraspAnyRegionDataset, collate_fn_sa2va_rl
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor
from projects.llava_sam2.rl_train.llm_reward_qa import LLMCaptionRewardQA
from projects.llava_sam2.rl_train.logits_processor import (
    NumericalStabilityLogitsProcessor,
    TemperatureLogitsWarper,
)
from projects.llava_sam2.hf.models.modeling_sa2va_chat import get_stop_criteria

from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList
from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel


class MaskCaptionRLTrainer:
    """
    Single-stage RL trainer for mask captioning.

    Task: Given image + mask region, generate a caption describing the region.
    Reward: LLM-based evaluation (via vLLM server with Llama-3.1-8B-Instruct).
    """

    def __init__(
        self,
        model,
        tokenizer,
        preprocessor,
        train_dataset,
        args,
        reward_model: Optional[LLMCaptionRewardQA] = None,
        device="cuda",
        num_generations=4,
        max_prompt_length=2048,
        max_completion_length=256,
        beta=0.01,
        # Length penalty parameters
        caption_min_length=15,
        caption_max_length=70,
        length_penalty_lambda=0.3,
        length_penalty_min_factor=0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.train_dataset = train_dataset
        self.args = args
        self.reward_model = reward_model
        self.device = device
        self.rank = 0  # Will be set in main() for multi-GPU

        self.num_generations = num_generations
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.beta = beta

        # Length penalty config
        self.caption_min_length = caption_min_length
        self.caption_max_length = caption_max_length
        self.length_penalty_lambda = length_penalty_lambda
        self.length_penalty_min_factor = length_penalty_min_factor

        # Generation config for mask->caption task
        self.generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=True,  # Required for GRPO diverse generations
            temperature=1.0,
            num_return_sequences=num_generations,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            max_length=None,  # Avoid conflicts with max_new_tokens
        )

        # Logits processors for numerical stability
        self.logits_processor = LogitsProcessorList([
            NumericalStabilityLogitsProcessor(clip_value=30.0, min_prob=1e-8, verbose=False),
            TemperatureLogitsWarper(temperature=1.0, min_temperature=0.1),
        ])

        # Stopping criteria
        stop_words = ['<|end|>']
        self.stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

        print("=" * 70)
        print("MaskCaptionRLTrainer initialized")
        print("=" * 70)
        print(f"  Reward model: {'LLM (' + reward_model.base_url + ')' if reward_model else 'None'}")
        print(f"  Num generations per sample: {self.num_generations}")
        print(f"  Max completion length: {self.max_completion_length}")
        print(f"  Caption min length: {self.caption_min_length}")
        print(f"  Caption max length: {self.caption_max_length}")
        print(f"  Length penalty lambda: {self.length_penalty_lambda}")

    def train_step(self, batch):
        """
        Execute one training step.

        Args:
            batch: Dict with keys: images, masks, captions (all lists)

        Returns:
            loss, metrics
        """
        images = batch["images"]
        gt_masks = batch["masks"]
        gt_captions = batch["captions"]
        batch_size = len(images)

        # Unwrap DDP model if needed
        model_to_use = self.model.module if hasattr(self.model, 'module') else self.model

        # Step 1: Prepare inputs for mask-to-caption generation
        preprocessed = []
        for img, mask, cap in zip(images, gt_masks, gt_captions):
            prep = self.preprocessor.prepare_for_model(
                image=img,
                mask=mask,
                caption=cap,
                task="mask_to_caption",
                instruction="Please describe this region in detail."
            )
            preprocessed.append(prep)

        # Prepare tensors
        pixel_values = torch.stack([p['pixel_values'] for p in preprocessed]).to(self.device)
        prompt_masks = [p['prompt_masks'].to(self.device) for p in preprocessed]
        vp_overall_mask = torch.stack([p['vp_overall_mask'] for p in preprocessed]).to(self.device).squeeze(-1)

        # Tokenize prompts
        prompt_texts = [p['prompt_text'] for p in preprocessed]
        prompt_texts = self._add_image_tokens(prompt_texts)

        # Apply template formatting
        template = model_to_use.template['INSTRUCTION']
        bot_name = getattr(model_to_use, 'bot_name', 'BOT')
        prompt_texts = [template.format(input=text, round=1, bot_name=bot_name) for text in prompt_texts]

        prompt_encodings = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )
        prompt_ids = prompt_encodings["input_ids"].to(self.device)
        prompt_mask = prompt_encodings["attention_mask"].to(self.device)

        # Calculate actual prompt lengths
        actual_prompt_lengths = prompt_mask.sum(dim=1)

        if self.max_prompt_length:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            actual_prompt_lengths = torch.clamp(actual_prompt_lengths, max=self.max_prompt_length)

        # Debug info
        if self.rank == 0:
            print(f"  [Debug] prompt_ids shape: {prompt_ids.shape}")
            print(f"  [Debug] actual_prompt_lengths: min={actual_prompt_lengths.min().item()}, max={actual_prompt_lengths.max().item()}")

        # Step 2: Generate captions (G samples per input)
        with torch.no_grad():
            output_ids = model_to_use.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                prompt_masks=prompt_masks,
                vp_overall_mask=vp_overall_mask,
                generation_config=self.generation_config,
                attention_mask=prompt_mask,
                logits_processor=self.logits_processor,
            )

        # Sa2VA's generate() returns only newly generated tokens
        completion_ids = output_ids

        if self.rank == 0:
            print(f"  [Debug] completion_ids shape: {completion_ids.shape}")

        # Decode completions
        pred_captions_raw = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=False)
        pred_captions = [self._clean_caption(cap) for cap in pred_captions_raw]

        if self.rank == 0:
            print(f"  [Debug] First 2 captions: {pred_captions[:2]}")

        # Step 3: Compute LLM rewards
        # Repeat GT captions for num_generations
        gt_captions_repeated = []
        for cap in gt_captions:
            gt_captions_repeated.extend([cap] * self.num_generations)

        # Use LLM reward model
        if self.reward_model is not None:
            llm_rewards = self.reward_model.compute_rewards(
                gt_captions_repeated, pred_captions, device=self.device
            )
        else:
            # Fallback to zero rewards if no reward model
            llm_rewards = torch.zeros(len(pred_captions), dtype=torch.float32, device=self.device)

        # Step 4: Apply length penalty
        caption_lengths = torch.tensor(
            [len(c.strip().split()) for c in pred_captions],
            dtype=torch.float32,
            device=self.device,
        )

        # Short caption penalty
        L_min = float(self.caption_min_length)
        short_gap = (L_min - caption_lengths).clamp(min=0.0)
        short_ratio = short_gap / max(L_min, 1.0)
        short_penalty = self.length_penalty_lambda * short_ratio
        short_factor = (1.0 - short_penalty).clamp(min=self.length_penalty_min_factor, max=1.0)

        # Long caption penalty
        L_max = float(self.caption_max_length)
        long_gap = (caption_lengths - L_max).clamp(min=0.0)
        long_ratio = (long_gap / max(L_max, 1.0)).clamp(max=1.0)
        long_penalty = 0.2 * long_ratio  # Mild penalty for long captions
        long_factor = (1.0 - long_penalty).clamp(min=0.8, max=1.0)

        # Combined length factor
        length_factor = (short_factor * long_factor).clamp(min=self.length_penalty_min_factor, max=1.0)

        # Final rewards
        rewards = llm_rewards * length_factor

        if self.rank == 0:
            print(f"  [Debug] LLM rewards: min={llm_rewards.min().item():.4f}, "
                  f"max={llm_rewards.max().item():.4f}, mean={llm_rewards.mean().item():.4f}")
            print(f"  [Debug] Caption lengths: min={caption_lengths.min().item():.1f}, "
                  f"max={caption_lengths.max().item():.1f}, mean={caption_lengths.mean().item():.1f}")
            print(f"  [Debug] Final rewards: min={rewards.min().item():.4f}, "
                  f"max={rewards.max().item():.4f}, mean={rewards.mean().item():.4f}")

        # Step 5: Compute GRPO loss
        # Repeat inputs for G generations
        prompt_ids_rep = prompt_ids.repeat_interleave(self.num_generations, dim=0)
        prompt_mask_rep = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        pixel_values_rep = pixel_values.repeat_interleave(self.num_generations, dim=0)
        prompt_masks_rep = [m for m in prompt_masks for _ in range(self.num_generations)]
        vp_overall_mask_rep = vp_overall_mask.repeat_interleave(self.num_generations, dim=0)

        loss, grpo_metrics = self._compute_grpo_loss(
            prompt_ids_rep, prompt_mask_rep, completion_ids,
            pixel_values_rep, prompt_masks_rep, vp_overall_mask_rep,
            rewards
        )

        # Combine metrics
        metrics = {
            'loss': loss.item(),
            'mean_llm_reward': llm_rewards.mean().item(),
            'mean_caption_len': caption_lengths.mean().item(),
            'mean_reward': rewards.mean().item(),
            **grpo_metrics,
        }

        return loss, metrics

    def _compute_grpo_loss(self, prompt_ids, prompt_mask, completion_ids,
                          pixel_values, prompt_masks, vp_overall_mask,
                          rewards):
        """
        Compute GRPO (Group Relative Policy Optimization) loss.
        """
        # Mask after EOS
        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            eos_positions = is_eos.int().argmax(dim=1)
            eos_idx[has_eos] = eos_positions[has_eos]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate for forward pass
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)

        # Forward pass
        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': input_ids.clone(),
            'pixel_values': pixel_values,
            'prompt_masks': prompt_masks,
            'vp_overall_mask': vp_overall_mask,
        }

        outputs = self.model(data, mode='loss')
        logits = outputs.logits[:, :-1, :]
        target_ids = input_ids[:, 1:]

        # Compute log probs with numerical stability
        logits = torch.clamp(logits, min=-100, max=100)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        token_log_probs = torch.nan_to_num(token_log_probs, nan=-100.0, posinf=0.0, neginf=-100.0)

        # Extract completion part
        prompt_length = prompt_ids.shape[1]
        completion_log_probs = token_log_probs[:, prompt_length-1:]

        # Group-wise advantage normalization
        batch_size = len(rewards) // self.num_generations
        rewards_grouped = rewards.view(batch_size, self.num_generations)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True)

        # Prevent division by zero
        epsilon = 1e-3
        std_rewards = torch.clamp(std_rewards, min=epsilon)

        advantages = (rewards_grouped - mean_rewards) / std_rewards
        advantages = advantages.view(-1)
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        if self.rank == 0:
            print(f"  [GRPO Debug] Advantages: min={advantages.min().item():.4f}, "
                  f"max={advantages.max().item():.4f}, mean={advantages.mean().item():.4f}")

        # Policy gradient loss
        per_token_loss = -completion_log_probs * advantages.unsqueeze(1)
        loss = (per_token_loss * completion_mask).sum() / (completion_mask.sum() + 1e-8)

        # Replace NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)

        metrics = {
            'reward_std': std_rewards.mean().item(),
            'completion_length': completion_mask.sum(1).float().mean().item(),
            'advantage_mean': advantages.mean().item(),
        }

        return loss, metrics

    def _add_image_tokens(self, texts):
        """Add image tokens to text."""
        IMG_TOKENS_PER_FRAME = 256
        image_token_str = f"<img>{'<IMG_CONTEXT>' * IMG_TOKENS_PER_FRAME}</img>"
        return [text.replace('<image>', image_token_str) for text in texts]

    def _clean_caption(self, caption):
        """Clean generated caption by removing special tokens."""
        import re

        text_output = caption.replace("<s>", "").replace("\n", "") \
            .replace("region1", '').replace("Region1", '') \
            .replace("The region marked by", "").replace("The region marked as", "").replace("The region marked", "") \
            .replace("is", "").replace("shows", "").replace(':', '') \
            .replace("   ", " ").replace("  ", " ")

        text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output)
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        return cleaned_str if cleaned_str else "object"


def main():
    parser = argparse.ArgumentParser(description="Sa2VA Mask Captioning RL Training with LLM Reward")
    parser.add_argument('--model_path', type=str, required=True, help='Path to pretrained Sa2VA model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to GAR dataset directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--num_generations', type=int, default=4, help='Number of generations per sample for GRPO')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--max_steps', type=int, default=-1, help='Max training steps (-1 for full epochs)')
    parser.add_argument('--save_steps', type=int, default=250, help='Save checkpoint every N steps')
    parser.add_argument('--log_steps', type=int, default=10, help='Log detailed metrics every N steps')
    parser.add_argument('--max_completion_length', type=int, default=256, help='Max tokens for caption generation')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    # LLM reward server configuration
    parser.add_argument('--llm_server_url', type=str, default='http://localhost:9100/v1',
                        help='vLLM server URL for LLM reward (OpenAI-compatible API)')
    parser.add_argument('--llm_model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                        help='LLM model name for reward evaluation')
    args = parser.parse_args()

    print("=" * 70)
    print("Sa2VA Mask Captioning RL Training")
    print("=" * 70)
    sys.stdout.flush()

    # Initialize distributed training
    import torch.distributed as dist
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        print(f"[Pre-init] Rank {global_rank}/{world_size} (local: {local_rank}) starting...")
        sys.stdout.flush()

        try:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            print(f"[Rank {global_rank}/{world_size}] Distributed training initialized on {device}")
            sys.stdout.flush()
        except Exception as e:
            print(f"[Rank {global_rank}] FATAL ERROR during init_process_group: {e}")
            sys.stdout.flush()
            import traceback
            traceback.print_exc()
            raise
    else:
        local_rank = 0
        world_size = 1
        device = "cuda"
        print("Single GPU training")
        sys.stdout.flush()

    # Initialize NLTK data
    print(f"\n[Rank {local_rank}] [1] Initializing NLTK data...")
    sys.stdout.flush()
    from projects.llava_sam2.rl_train.reward_functions import _ensure_nltk_data
    _ensure_nltk_data()
    print(f"[Rank {local_rank}] NLTK data ready")
    sys.stdout.flush()

    # Load dataset
    print(f"\n[Rank {local_rank}] [2] Loading dataset...")
    sys.stdout.flush()
    dataset = GraspAnyRegionDataset(local_data_dir=args.data_dir, parts_to_load=None)
    print(f"[Rank {local_rank}] Loaded {len(dataset)} samples")
    sys.stdout.flush()

    # Load model
    print(f"\n[Rank {local_rank}] [3] Loading model...")
    sys.stdout.flush()

    model = Sa2VAChatModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print(f"[Rank {local_rank}] Model loaded")
    sys.stdout.flush()

    # Apply LoRA to language model
    print(f"\n[Rank {local_rank}] [4] Applying LoRA and freezing components...")
    sys.stdout.flush()
    model.wrap_llm_lora(r=128, lora_alpha=256, lora_dropout=0.05)
    model.to(device)

    # ========================================
    # Freeze/Unfreeze configuration (matching SFT training)
    # ========================================

    # 1. Freeze vision encoder (InternViT)
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print(f"[Rank {local_rank}]   - Vision encoder: FROZEN")

    # 2. Freeze SAM2 encoder, but keep mask decoder trainable
    # grounding_encoder contains: image_encoder, memory_encoder, sam_mask_decoder
    model.grounding_encoder.requires_grad_(False)  # Freeze all first
    # Then unfreeze mask decoder
    if hasattr(model.grounding_encoder, 'sam2_model'):
        model.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)
        print(f"[Rank {local_rank}]   - SAM2 encoder: FROZEN")
        print(f"[Rank {local_rank}]   - SAM2 mask decoder: TRAINABLE")
    else:
        print(f"[Rank {local_rank}]   - SAM2: FROZEN (no sam2_model found)")

    # 3. Projectors are trainable by default (mlp1, text_hidden_fcs)
    # Ensure they are trainable
    for param in model.mlp1.parameters():
        param.requires_grad = True
    for param in model.text_hidden_fcs.parameters():
        param.requires_grad = True
    print(f"[Rank {local_rank}]   - Vision projector (mlp1): TRAINABLE")
    print(f"[Rank {local_rank}]   - Text projector (text_hidden_fcs): TRAINABLE")

    # 4. Language model: LoRA already applied, base model frozen
    print(f"[Rank {local_rank}]   - Language model: LoRA TRAINABLE")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Rank {local_rank}]   Total params: {total_params:,}")
    print(f"[Rank {local_rank}]   Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    sys.stdout.flush()

    # Prepare for generation
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.preparing_for_generation(tokenizer, max_new_tokens=args.max_completion_length)

    # Update generation configs
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_new_tokens = args.max_completion_length

    if hasattr(model, 'language_model') and hasattr(model.language_model, 'generation_config'):
        if model.language_model.generation_config is not None:
            model.language_model.generation_config.max_new_tokens = args.max_completion_length

    print(f"[Rank {local_rank}] Model prepared for generation")
    sys.stdout.flush()

    # Wrap in DDP
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        model._set_static_graph()
        print(f"[Rank {local_rank}] Model wrapped in DDP")
        sys.stdout.flush()

    # Initialize preprocessor
    preprocessor = Sa2VADataPreprocessor()

    # Initialize LLM reward model (QA-style evaluation like DLC-Bench)
    print(f"\n[Rank {local_rank}] [5] Initializing LLM reward model (QA-style)...")
    sys.stdout.flush()

    reward_model = LLMCaptionRewardQA(
        base_url=args.llm_server_url,
        model=args.llm_model,
        verbose=(local_rank == 0),
        min_questions=2,
        max_questions=5,
    )
    print(f"[Rank {local_rank}]   LLM server: {args.llm_server_url}")
    print(f"[Rank {local_rank}]   LLM model: {args.llm_model}")
    print(f"[Rank {local_rank}]   Eval mode: QA (DLC-Bench style)")

    # Verify LLM server connection before training (only rank 0)
    if local_rank == 0:
        print(f"[Rank {local_rank}]   Verifying LLM server connection...")
        try:
            reward_model.verify_connection()
            print(f"[Rank {local_rank}]   LLM server connection: OK")
        except ConnectionError as e:
            print(f"\n{'='*70}")
            print("ERROR: LLM server is not available!")
            print('='*70)
            print(str(e))
            print('='*70)
            sys.stdout.flush()
            if world_size > 1:
                dist.destroy_process_group()
            sys.exit(1)
    sys.stdout.flush()

    # Create trainer
    print(f"\n[Rank {local_rank}] [6] Creating trainer...")
    sys.stdout.flush()
    trainer = MaskCaptionRLTrainer(
        model=model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        train_dataset=dataset,
        args=args,
        reward_model=reward_model,
        device=device,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
    )
    trainer.rank = local_rank

    # Create optimizer
    print(f"\n[Rank {local_rank}] [7] Setting up optimizer...")
    sys.stdout.flush()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    # Create dataloader
    from torch.utils.data import DataLoader, DistributedSampler

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=local_rank,
            shuffle=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn_sa2va_rl,
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=collate_fn_sa2va_rl,
            shuffle=True,
        )

    model.train()
    steps_per_epoch = len(dataloader)

    if local_rank == 0:
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Num epochs: {args.num_epochs}")
        print(f"  Num generations: {args.num_generations}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Save every: {args.save_steps} steps")
        print(f"{'='*70}\n")

    # Initialize TensorBoard
    writer = None
    if local_rank == 0:
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"TensorBoard logging: {tensorboard_dir}\n")

    global_step = 0
    llm_reward_history = []
    reward_history = []

    # Training loop
    print(f"\n[Rank {local_rank}] [8] Starting training...")
    sys.stdout.flush()

    for epoch in range(args.num_epochs):
        if local_rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'='*70}\n")

        if world_size > 1 and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        for step_in_epoch, batch in enumerate(dataloader):
            # Check max_steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            # Training step
            loss, metrics = trainer.train_step(batch)

            # Track history
            llm_reward_history.append(metrics.get('mean_llm_reward', 0))
            reward_history.append(metrics.get('mean_reward', 0))

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Clear cache
            torch.cuda.empty_cache()

            # Console logging
            if global_step % 1 == 0:
                print(f"[Rank {local_rank}] Epoch {epoch+1} Step {step_in_epoch}/{steps_per_epoch}: "
                      f"loss={loss.item():.4f}, llm_reward={metrics.get('mean_llm_reward', 0):.4f}, "
                      f"reward={metrics.get('mean_reward', 0):.4f}")

            # TensorBoard logging
            if local_rank == 0 and writer is not None:
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Reward/llm', metrics.get('mean_llm_reward', 0), global_step)
                writer.add_scalar('Reward/final', metrics.get('mean_reward', 0), global_step)
                writer.add_scalar('Metrics/caption_length', metrics.get('mean_caption_len', 0), global_step)
                writer.add_scalar('Metrics/completion_length', metrics.get('completion_length', 0), global_step)
                writer.add_scalar('Metrics/reward_std', metrics.get('reward_std', 0), global_step)
                writer.flush()

            # Detailed logging
            if local_rank == 0 and global_step > 0 and global_step % args.log_steps == 0:
                print(f"\n{'='*70}")
                print(f"DETAILED METRICS @ Step {global_step}")
                print(f"{'='*70}")
                if len(llm_reward_history) > 0:
                    recent = llm_reward_history[-args.log_steps:]
                    print(f"LLM Reward (last {len(recent)} batches):")
                    print(f"  Min:  {min(recent):.4f}")
                    print(f"  Max:  {max(recent):.4f}")
                    print(f"  Mean: {sum(recent)/len(recent):.4f}")
                if len(reward_history) > 0:
                    recent = reward_history[-args.log_steps:]
                    print(f"Final Reward (last {len(recent)} batches):")
                    print(f"  Min:  {min(recent):.4f}")
                    print(f"  Max:  {max(recent):.4f}")
                    print(f"  Mean: {sum(recent)/len(recent):.4f}")
                print(f"{'='*70}\n")

            # Save checkpoint
            if local_rank == 0 and global_step > 0 and global_step % args.save_steps == 0:
                print(f"\n{'='*70}")
                print(f"SAVING CHECKPOINT @ Step {global_step}")
                print(f"{'='*70}")
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_dir = f"{args.output_dir}/checkpoint-{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                model_to_save.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved to {checkpoint_dir}")
                print(f"{'='*70}\n")

            global_step += 1

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

        if local_rank == 0:
            print(f"\n{'='*70}")
            print(f"Epoch {epoch + 1}/{args.num_epochs} completed ({step_in_epoch + 1} steps)")
            print(f"{'='*70}\n")

    # Final save
    print(f"\n{'='*70}")
    print("Training completed!")
    print(f"{'='*70}")

    if local_rank == 0 and writer is not None:
        writer.close()
        print("TensorBoard writer closed")

    if local_rank == 0:
        model_to_save = model.module if hasattr(model, 'module') else model
        final_path = f"{args.output_dir}/final_model"
        os.makedirs(final_path, exist_ok=True)
        model_to_save.save_pretrained(final_path)
        print(f"Final model saved to {final_path}")

    # Cleanup distributed
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
