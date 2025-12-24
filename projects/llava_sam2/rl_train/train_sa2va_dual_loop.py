"""
Sa2VA Dual-Loop RL Training - Complete Implementation

Loop 1: mask → caption → mask'
Loop 2: caption → mask → caption'

Both loops train simultaneously on the same batch.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from typing import List
import copy
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from projects.llava_sam2.rl_train.dataset_gar import GraspAnyRegionDataset, collate_fn_sa2va_rl
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor
from projects.llava_sam2.rl_train.reward_functions import iou_reward_batch, combined_caption_reward
from projects.llava_sam2.rl_train.training_callbacks import GradientMonitorCallback
from projects.llava_sam2.rl_train.logits_processor import (
    NumericalStabilityLogitsProcessor,
    TemperatureLogitsWarper,
)
from transformers import StoppingCriteriaList, StoppingCriteria
from projects.llava_sam2.hf.models.modeling_sa2va_chat import get_stop_criteria

from trl import GRPOConfig
from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList
from peft import LoraConfig

from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    """Extract hidden states for [SEG] tokens."""
    seg_mask = output_ids == seg_id
    n_out = seg_mask.sum().item()
    if n_out == 0:
        return hidden_states[0:0]
    # Find positions of [SEG] tokens
    return hidden_states[-n_out:][seg_mask]


def compute_iou_batch(pred_masks, gt_masks):
    """
    Compute IoU between predicted and GT masks.

    Args:
        pred_masks: (B, H, W) boolean or float
        gt_masks: (B, H, W) boolean or float

    Returns:
        ious: (B,) tensor
    """
    pred_bool = pred_masks > 0.5 if pred_masks.dtype == torch.float else pred_masks
    gt_bool = gt_masks > 0.5 if gt_masks.dtype == torch.float else gt_masks

    intersection = (pred_bool & gt_bool).float().sum(dim=(1, 2))
    union = (pred_bool | gt_bool).float().sum(dim=(1, 2))
    iou = intersection / (union + 1e-6)
    return iou


class DualLoopRLTrainer:
    """
    Complete dual-loop RL trainer for Sa2VA.

    This trainer implements both Loop 1 and Loop 2 simultaneously.
    """

    def __init__(
        self,
        model,
        ema_model,
        tokenizer,
        preprocessor,
        train_dataset,
        args,
        device="cuda",
        num_generations=2,
        max_prompt_length=2048,
        max_completion_length=256,  # Reduced for Loop 1 (do_sample=True) to prevent OOM
                                     # Loop 2 (EMA) will still use 2048 (do_sample=False)
        beta=0.01,
        loop1_weight=1.0,
        loop2_weight=0.0,
        ema_decay=0.999,
        # ===== 新增：长度惩罚相关超参 =====
        caption_min_length=25,          # 低于这个词数开始被惩罚
        length_penalty_lambda=0.5,      # 惩罚强度（越大惩罚越狠）
        length_penalty_min_factor=0.25,  # reward 最低被打到多少（乘法系数下界）
        length_penalty_power=1.0,       # >1 会让特别短的被更狠打击
    ):
        self.model = model
        self.ema_model = ema_model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.train_dataset = train_dataset
        self.args = args
        self.device = device
        self.rank = 0  # Default to 0 for single GPU, will be set in main() for multi-GPU

        self.num_generations = num_generations
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.beta = beta
        self.loop1_weight = loop1_weight
        self.loop2_weight = loop2_weight
        self.ema_decay = ema_decay
        # ===== 保存长度惩罚配置 =====
        self.caption_min_length = caption_min_length
        self.length_penalty_lambda = length_penalty_lambda
        self.length_penalty_min_factor = length_penalty_min_factor
        self.length_penalty_power = length_penalty_power

        # Generation config for mask→caption tasks
        # CRITICAL: Only set max_new_tokens (like SFT training), do NOT set max_length
        # Setting both causes conflicts in transformers.generate()
        self.generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,  # Number of NEW tokens to generate
            do_sample=True,  # Required for GRPO diverse generations
            temperature=1.0,  # Same as original, allows longer/more diverse generation
            num_return_sequences=num_generations,
            eos_token_id=tokenizer.eos_token_id,  # CRITICAL: Must set EOS token
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            max_length=None,  # CRITICAL: Set to None to avoid conflicts with max_new_tokens
        )

        # Logits processors for numerical stability
        self.logits_processor = LogitsProcessorList([
            NumericalStabilityLogitsProcessor(clip_value=30.0, min_prob=1e-8, verbose=False),
            TemperatureLogitsWarper(temperature=1.0, min_temperature=0.1),
        ])

        # Stopping criteria (same as Sa2VA SFT training)
        # phi3_chat template has stop word: '<|end|>'
        stop_words = ['<|end|>']
        self.stop_criteria = get_stop_criteria(tokenizer=tokenizer, stop_words=stop_words)

        # Get [SEG] token ID
        self.seg_token_id = tokenizer.convert_tokens_to_ids('[SEG]')

        print("✓ DualLoopRLTrainer initialized")
        print(f"  Loop 1 weight: {self.loop1_weight}")
        print(f"  Loop 2 weight: {self.loop2_weight}")
        print(f"  EMA decay: {self.ema_decay}")
        print(f"  Num generations: {self.num_generations}")
        print(f"  [SEG] token ID: {self.seg_token_id}")

    def train_step(self, batch, compute_separate_grads=False):
        """
        Execute one training step with both loops.

        Args:
            batch: Dict with keys: images, masks, captions (all lists)
            compute_separate_grads: If True, compute Loop1 and Loop2 gradients separately for analysis

        Returns:
            loss, metrics, (loop1_grads, loop2_grads) if compute_separate_grads else None
        """
        images = batch["images"]
        gt_masks = batch["masks"]
        gt_captions = batch["captions"]
        batch_size = len(images)

        # ============================================================
        # LOOP 1: mask → caption → mask'
        # ============================================================

        loss_loop1, metrics_loop1 = self._loop1_mask_to_caption_to_mask(
            images, gt_masks, gt_captions
        )

        # ============================================================
        # LOOP 2: caption → mask → caption'
        # ============================================================
        '''
        loss_loop2, metrics_loop2 = self._loop2_caption_to_mask_to_caption(
            images, gt_masks, gt_captions
        )
        '''
        # 临时禁用 Loop 2，专注调试 Loop 1
        loss_loop2 = torch.zeros_like(loss_loop1)
        metrics_loop2 = {}
        
        # ============================================================
        # Combine losses
        # ============================================================

        #total_loss = self.loop1_weight * loss_loop1 + self.loop2_weight * loss_loop2
        total_loss = loss_loop1

        # CRITICAL FIX: Final safeguard against NaN in total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=total_loss.device, dtype=total_loss.dtype, requires_grad=True)

        # Update EMA model
        self._update_ema_model()

        # Combine metrics
        metrics = {
            **{f"loop1/{k}": v for k, v in metrics_loop1.items()},
            **{f"loop2/{k}": v for k, v in metrics_loop2.items()},
            "loop1_loss": loss_loop1.item(),
            "loop2_loss": loss_loop2.item(),
            "total_loss": total_loss.item(),
        }

        # Optionally compute separate gradients for analysis
        separate_grads = None
        '''
        if compute_separate_grads:
            separate_grads = self._compute_separate_gradients(loss_loop1, loss_loop2)
            self.model.zero_grad(set_to_none=True)
        '''

        return total_loss, metrics, separate_grads

    def _loop1_mask_to_caption_to_mask(self, images, gt_masks, gt_captions):
        """
        Loop 1: mask → caption → mask'

        1. image + GT_mask → model generates caption (G samples)
        2. image + pred_caption → EMA model generates mask'
        3. Reward: IoU(mask', GT_mask)
        4. GRPO loss
        """
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

        # CRITICAL FIX: Apply template formatting (same as predict_forward)
        # phi3_chat template: '<|user|>\n{input}<|end|>\n<|assistant|>\n'
        template = model_to_use.template['INSTRUCTION']
        bot_name = getattr(model_to_use, 'bot_name', 'BOT')
        prompt_texts = [template.format(input=text, round=1, bot_name=bot_name) for text in prompt_texts]

        prompt_encodings = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",  # Left padding for generation
            add_special_tokens=True,
        )
        prompt_ids = prompt_encodings["input_ids"].to(self.device)
        prompt_mask = prompt_encodings["attention_mask"].to(self.device)

        # Calculate actual prompt lengths (excluding left padding)
        # attention_mask: 1 for real tokens, 0 for padding
        actual_prompt_lengths = prompt_mask.sum(dim=1)  # (batch_size,)

        if self.max_prompt_length:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            # Update actual_prompt_lengths after truncation
            actual_prompt_lengths = torch.clamp(actual_prompt_lengths, max=self.max_prompt_length)

        # Debug: Show prompt info before generation
        if self.rank == 0:
            print(f"  [Loop1 Debug] prompt_ids shape: {prompt_ids.shape}")
            print(f"  [Loop1 Debug] actual_prompt_lengths (min/max/mean): {actual_prompt_lengths.min().item()}/{actual_prompt_lengths.max().item()}/{actual_prompt_lengths.float().mean().item():.1f}")
            print(f"  [Loop1 Debug] generation_config.max_new_tokens: {self.generation_config.max_new_tokens}")
            print(f"  [Loop1 Debug] generation_config.max_length: {self.generation_config.max_length if hasattr(self.generation_config, 'max_length') else 'Not set'}")

        # Generate captions (G samples per input)
        # CRITICAL: Do NOT use stopping_criteria when num_return_sequences > 1
        # Because StopWordStoppingCriteria only checks input_ids[0], causing other sequences to be truncated!
        # Let model naturally generate to EOS token, then clean <|end|> in post-processing
        with torch.no_grad():
            output_ids = model_to_use.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                prompt_masks=prompt_masks,
                vp_overall_mask=vp_overall_mask,
                generation_config=self.generation_config,
                attention_mask=prompt_mask,
                logits_processor=self.logits_processor,
                # stopping_criteria=self.stop_criteria,  # REMOVED: Causes truncation with num_return_sequences > 1
            )

        # Debug: Check generation results
        if self.rank == 0:
            print(f"  [Loop1 Debug] output_ids shape: {output_ids.shape}")
            print(f"  [Loop1 Debug] prompt_ids shape: {prompt_ids.shape}")

        # CRITICAL UNDERSTANDING:
        # Sa2VA's generate() uses inputs_embeds (not input_ids)
        # When using inputs_embeds, transformers ONLY returns newly generated tokens
        # output_ids does NOT contain the prompt part!
        # So output_ids IS the completion_ids!
        completion_ids = output_ids

        if self.rank == 0:
            print(f"  [Loop1 Debug] completion_ids shape: {completion_ids.shape}")
            full_text = self.tokenizer.decode(completion_ids[0], skip_special_tokens=False) if completion_ids.shape[1] > 0 else 'EMPTY'
            print(f"  [Loop1 Debug] First completion (full): {full_text}")
            print(f"  [Loop1 Debug] First completion (length in chars): {len(full_text)}")

        # Decode completions to get captions
        pred_captions_raw = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=False)

        # CRITICAL FIX: Clean captions using the same logic as Loop 2
        pred_captions = []
        for cap in pred_captions_raw:
            cleaned = self._clean_caption(cap)
            pred_captions.append(cleaned)

        # Debug: Show captions before and after cleaning
        if self.rank == 0:
            print(f"  [Loop1 Debug] First raw caption (full): {pred_captions_raw[0] if pred_captions_raw else 'EMPTY'}")
            print(f"  [Loop1 Debug] First cleaned caption (full): {pred_captions[0] if pred_captions else 'EMPTY'}")

        # Step 2: Use EMA model to generate masks from pred_captions
        # For each generated caption, generate mask using EMA model
        pred_masks_list = []
        for i in range(len(pred_captions)):
            original_idx = i // self.num_generations
            caption = pred_captions[i]
            image = images[original_idx]
            gt_mask = gt_masks[original_idx]  # Get GT mask for size reference

            # Generate mask from caption using EMA model
            pred_mask = self._generate_mask_from_caption_ema(image, caption, gt_mask)
            pred_masks_list.append(pred_mask)

        # Use _stack_masks_with_resize to handle different sizes (fixes batch_size > 1 bug)
        pred_masks = self._stack_masks_with_resize(pred_masks_list, self.device)  # (B*G, H, W)

        # Step 3: Compute IoU rewards
        gt_masks_repeated = self._repeat_masks(gt_masks, self.num_generations)  # (B*G, H, W)
        iou_rewards = compute_iou_batch(pred_masks, gt_masks_repeated)  # (B*G,)

        # ===== 新增：对“太短 caption”做软惩罚（< caption_min_length） =====
        # 1) 计算每个 caption 的词数（B*G）
        caption_lengths = torch.tensor(
            [len(c.strip().split()) for c in pred_captions],
            dtype=torch.float32,
            device=self.device,
        )  # (B*G,)

        L_min = float(self.caption_min_length)

        # 2) 对低于 L_min 的部分计算“短了多少”
        #    short_gap > 0 说明比 L_min 短；>= L_min 时 gap = 0（不惩罚）
        short_gap = (L_min - caption_lengths).clamp(min=0.0)  # (B*G,)
        # 归一化到 [0, 1]，再可选地做个幂次（power > 1 会让特别短的更惨一点）
        short_ratio = (short_gap / max(L_min, 1.0)) ** self.length_penalty_power  # (B*G,)

        # 3) 惩罚值：越短 short_ratio 越大，惩罚越大
        penalty = self.length_penalty_lambda * short_ratio  # (B*G,)

        # 4) 转成乘法系数（1 是不惩罚；越短越接近 length_penalty_min_factor）
        length_factor = (1.0 - penalty).clamp(
            min=self.length_penalty_min_factor,
            max=1.0,
        )  # (B*G,)

        # ===== 新增：对“过长 caption”做轻微惩罚（> 70 个词） =====
        L_max = 120.0
        long_gap = (caption_lengths - L_max).clamp(min=0.0)  # 超出多少词
        # 归一化到 [0, 1]，最多视为“超出 100%”
        long_ratio = (long_gap / max(L_max, 1.0)).clamp(max=1.0)

        # 超长惩罚强度（不要太大，不然模型会突然变得过短）
        long_penalty_lambda = 0.3  # 你可以以后再调
        long_penalty = long_penalty_lambda * long_ratio  # (B*G,)

        # 转成乘法系数，和短句的 length_factor 叠乘
        long_factor_min = 0.6  # 最多把 reward 打到 0.7
        long_factor = (1.0 - long_penalty).clamp(min=long_factor_min, max=1.0)

        # 短惩罚 * 长惩罚，两种都考虑
        length_factor = (length_factor * long_factor).clamp(
            min=self.length_penalty_min_factor,
            max=1.0,
        )        

        # 5) 最终 reward：IoU * length_factor
        rewards = iou_rewards * length_factor

        # Debug: IoU + 长度信息
        if self.rank == 0:
            print(f"  [Loop1 Debug] IoU rewards: min={iou_rewards.min().item():.4f}, "
                  f"max={iou_rewards.max().item():.4f}, mean={iou_rewards.mean().item():.4f}, "
                  f"std={iou_rewards.std().item():.4f}")
            print(f"[Loop1 Debug] output_ids shape: {output_ids.shape}")
            print(f"[Loop1 Debug] completion_ids shape: {completion_ids.shape}")
            print(f"[Loop1 Debug] first 1-2 captions: {pred_captions[:2]}")
            print(f"  [Loop1 Debug] Caption lengths: "
                  f"min={caption_lengths.min().item():.1f}, "
                  f"max={caption_lengths.max().item():.1f}, "
                  f"mean={caption_lengths.mean().item():.1f}")
            print(f"  [Loop1 Debug] short_ratio: "
                  f"min={short_ratio.min().item():.4f}, "
                  f"max={short_ratio.max().item():.4f}, "
                  f"mean={short_ratio.mean().item():.4f}")
            print(f"  [Loop1 Debug] long_ratio: "
                  f"min={long_ratio.min().item():.4f}, "
                  f"max={long_ratio.max().item():.4f}, "
                  f"mean={long_ratio.mean().item():.4f}")
            print(f"  [Loop1 Debug] length_factor: "
                  f"min={length_factor.min().item():.4f}, "
                  f"max={length_factor.max().item():.4f}, "
                  f"mean={length_factor.mean().item():.4f}")
            print(f"  [Loop1 Debug] Final rewards (with length penalty): "
                  f"min={rewards.min().item():.4f}, "
                  f"max={rewards.max().item():.4f}, "
                  f"mean={rewards.mean().item():.4f}")
        # ===== 新增结束 =====

        # Step 4: Compute GRPO loss
        # Re-generate with gradients to get log probs
        # Repeat inputs for G generations
        prompt_ids_rep = prompt_ids.repeat_interleave(self.num_generations, dim=0)
        prompt_mask_rep = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        pixel_values_rep = pixel_values.repeat_interleave(self.num_generations, dim=0)
        prompt_masks_rep = [m for m in prompt_masks for _ in range(self.num_generations)]
        vp_overall_mask_rep = vp_overall_mask.repeat_interleave(self.num_generations, dim=0)

        # Compute log probs
        # Pass prompt_ids length for GRPO loss calculation
        loss, metrics = self._compute_grpo_loss(
            prompt_ids_rep, prompt_mask_rep, completion_ids,
            pixel_values_rep, prompt_masks_rep, vp_overall_mask_rep,
            rewards
        )

        metrics['mean_iou'] = iou_rewards.mean().item()
        metrics['mean_caption_len'] = caption_lengths.mean().item()

        return loss, metrics

    def _loop2_caption_to_mask_to_caption(self, images, gt_masks, gt_captions):
        """
        Loop 2: caption → mask → caption'

        1. image + GT_caption → model generates mask ([SEG] token) WITH GRADIENTS
        2. image + pred_mask → EMA model generates caption'
        3. Reward: METEOR(caption', GT_caption)
        4. Reward-weighted supervised mask loss (BCE + Dice)
        """
        batch_size = len(images)

        # Step 1: Generate masks from GT captions (with gradients)
        pred_masks_list = []
        pred_masks_raw_list = []  # Raw logits for loss computation

        for img, gt_cap, gt_mask in zip(images, gt_captions, gt_masks):
            # Generate mask WITH gradients using forward pass
            pred_mask, pred_mask_raw = self._generate_mask_from_caption_with_grad(img, gt_cap, gt_mask)
            pred_masks_list.append(pred_mask)
            pred_masks_raw_list.append(pred_mask_raw)

        # Use _stack_masks_with_resize to handle different sizes (fixes batch_size > 1 bug)
        pred_masks = self._stack_masks_with_resize(pred_masks_list, self.device)  # (B, H, W)
        pred_masks_raw = self._stack_masks_with_resize(pred_masks_raw_list, self.device)  # (B, H, W) - logits

        # Step 2: Use EMA model to generate captions from pred_masks (detach for inference)
        pred_captions_list = []
        for img, pred_mask in zip(images, pred_masks):
            pred_caption = self._generate_caption_from_mask_ema(img, pred_mask.detach())
            pred_captions_list.append(pred_caption)

        # Step 3: Compute METEOR rewards
        meteor_rewards = self._compute_meteor_rewards(pred_captions_list, gt_captions)

        # Debug output
        if self.rank == 0:
            print(f"  [Loop2 Debug] METEOR rewards: min={meteor_rewards.min().item():.4f}, "
                  f"max={meteor_rewards.max().item():.4f}, mean={meteor_rewards.mean().item():.4f}")
            print(f"  [Loop2 Debug] Generated captions: {pred_captions_list[:2]}")

        # Step 4: Compute reward-weighted supervised mask loss
        # Prepare GT masks - convert to tensors
        gt_masks_list = []
        for gt_mask in gt_masks:
            if isinstance(gt_mask, torch.Tensor):
                gt_mask_tensor = gt_mask.float()
            else:
                import numpy as np
                gt_mask_np = np.array(gt_mask)
                gt_mask_tensor = torch.from_numpy(gt_mask_np).float()

            gt_masks_list.append(gt_mask_tensor)

        # Use _stack_masks_with_resize to handle different sizes (fixes batch_size > 1 bug)
        # Match the size of pred_masks_raw
        target_size = pred_masks_raw.shape[-2:]  # (H, W) from pred_masks_raw
        gt_masks_tensor = self._stack_masks_with_resize(gt_masks_list, self.device, target_size=target_size)  # (B, H, W)

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks_raw, gt_masks_tensor, reduction='none'
        )
        bce_loss = bce_loss.mean(dim=[1, 2])  # (B,)

        # Compute Dice loss
        pred_masks_sigmoid = torch.sigmoid(pred_masks_raw)
        dice_loss = self._compute_dice_loss(pred_masks_sigmoid, gt_masks_tensor)  # (B,)

        # Combine losses
        mask_loss = bce_loss + dice_loss  # (B,)

        # Weight by rewards: higher reward = lower weight (we want to reduce loss for good masks)
        # Normalize rewards to [0, 1] range for stable weighting
        # CRITICAL FIX: Prevent NaN when all rewards are identical
        reward_range = meteor_rewards.max() - meteor_rewards.min()
        if reward_range < 1e-6:
            # All rewards are the same - use uniform weights
            meteor_rewards_normalized = torch.zeros_like(meteor_rewards)
        else:
            meteor_rewards_normalized = (meteor_rewards - meteor_rewards.min()) / (reward_range + 1e-8)
        loss_weights = 1.0 - meteor_rewards_normalized * 0.5  # Range: [0.5, 1.0]

        # Final loss: weighted average
        weighted_loss = (mask_loss * loss_weights).mean()

        # Small weight to balance with Loop 1
        loss = weighted_loss * 0.2

        # CRITICAL FIX: Replace NaN/Inf loss with zero to prevent training collapse
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)

        metrics = {
            'mean_meteor': meteor_rewards.mean().item(),
            'reward': meteor_rewards.mean().item(),
            'bce_loss': bce_loss.mean().item(),
            'dice_loss': dice_loss.mean().item(),
        }

        return loss, metrics

    def _generate_mask_from_caption_ema(self, image, caption, gt_mask):
        """
        Use EMA model to generate mask from caption (Referring Expression Segmentation).

        Args:
            image: PIL Image
            caption: str (referring expression)
            gt_mask: PIL Image or tensor (for getting correct size)

        Returns:
            mask: (H, W) tensor
        """
        import numpy as np

        try:
            # Format text for RES (custom template for RL training)
            text_for_res = f'<image>\nThis is a description for something in the image: "{caption}", please segment it in the image.'

            # Use Sa2VA's predict_forward for RES inference
            with torch.no_grad():
                out = self.ema_model.predict_forward(
                    image=image,
                    text=text_for_res,
                    tokenizer=self.tokenizer,
                    return_logprobs=False
                )

            pred_masks = out.get('prediction_masks', None)

            if pred_masks is None or len(pred_masks) == 0:
                # Fallback: return empty mask with correct size
                if isinstance(gt_mask, torch.Tensor):
                    h, w = gt_mask.shape[-2:]
                else:
                    mask_array = np.array(gt_mask)
                    h, w = mask_array.shape[:2]
                if self.rank == 0:
                    print(f"⚠ WARNING [Loop1]: EMA model returned no masks for caption: '{caption[:50]}...'")
                return torch.zeros((h, w), dtype=torch.float32, device=self.device)

            # Take first mask (Sa2VA returns list of masks for multiple [SEG] tokens)
            first_mask = pred_masks[0]

            # Convert to tensor if needed
            if isinstance(first_mask, np.ndarray):
                mask_tensor = torch.from_numpy(first_mask).float()
            elif isinstance(first_mask, torch.Tensor):
                mask_tensor = first_mask.float()
            else:
                raise TypeError(f"Unexpected mask type: {type(first_mask)}")

            # Ensure 2D and move to correct device
            if mask_tensor.ndim > 2:
                mask_tensor = mask_tensor.squeeze()

            return mask_tensor.to(self.device)

        except Exception as e:
            # CRITICAL: Catch any exception to prevent training crash
            if self.rank == 0:
                print(f"\n{'='*70}")
                print(f"⚠ ERROR [Loop1]: EMA model failed to generate mask!")
                print(f"  Caption: '{caption[:100]}...'")
                print(f"  Error: {type(e).__name__}: {str(e)}")
                print(f"  Returning zero mask to continue training")
                print(f"{'='*70}\n")

            # Return zero mask with correct size
            if isinstance(gt_mask, torch.Tensor):
                h, w = gt_mask.shape[-2:]
            else:
                import numpy as np
                mask_array = np.array(gt_mask)
                h, w = mask_array.shape[:2]

            return torch.zeros((h, w), dtype=torch.float32, device=self.device)

    def _generate_mask_from_caption_with_grad(self, image, caption, gt_mask):
        """
        Generate mask from caption WITH GRADIENTS for Loop 2 RL training.

        This follows the original Sa2VA training forward pass:
        1. Construct input with [SEG] token (teacher forcing)
        2. Forward through model to get hidden states
        3. Extract [SEG] token embeddings
        4. Generate mask through SAM2 decoder (differentiable)

        Args:
            image: PIL Image
            caption: str (referring expression)
            gt_mask: PIL Image or tensor (for getting correct size)

        Returns:
            mask_binary: (H, W) tensor - binary mask for reward computation
            mask_logits: (H, W) tensor - raw mask logits (before sigmoid) for loss
        """
        import numpy as np
        from projects.llava_sam2.hf.models.modeling_sa2va_chat import dynamic_preprocess

        # Unwrap DDP if needed
        if hasattr(self.model, 'module'):
            model_to_use = self.model.module
        else:
            model_to_use = self.model

        # Prepare image inputs (similar to predict_forward)
        ori_image_size = image.size

        # Grounding image
        g_image = np.array(image)
        g_image = model_to_use.extra_image_processor.apply_image(g_image)
        g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(model_to_use.torch_dtype)
        g_pixel_values = torch.stack([
            model_to_use.grounding_encoder.preprocess_image(g_pixel_values)
        ]).to(self.device)

        # Vision encoder image
        images = dynamic_preprocess(image, model_to_use.min_dynamic_patch,
                                    model_to_use.max_dynamic_patch,
                                    model_to_use.image_size, model_to_use.use_thumbnail)
        pixel_values = [model_to_use.transformer(img) for img in images]
        pixel_values = torch.stack(pixel_values).to(model_to_use.torch_dtype).to(self.device)
        num_image_tokens = pixel_values.shape[0] * model_to_use.patch_token

        # Construct text with [SEG] token (teacher forcing)
        image_token_str = f'{model_to_use.IMG_START_TOKEN}' \
                          f'{model_to_use.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{model_to_use.IMG_END_TOKEN}\n'

        # Add [SEG] token after the caption (custom template for RL training)
        #text_for_res = f'<image>\nThis is a description for something in the image: "{caption}", please segment it in the image. [SEG]'
        #text_for_res = text_for_res.replace('<image>', image_token_str)
        # Question (aligned with ReVOS SFT training - no [SEG] in question)
        question = f'<image>\nThis is a description for something in the image: "{caption}", please segment it in the image.'
        question = question.replace('<image>', image_token_str)
        # Format question with template
        question_formatted = model_to_use.template['INSTRUCTION'].format(
            input=question, round=1, bot_name=model_to_use.bot_name)
        # Answer (aligned with ReVOS SFT training - [SEG] in answer)
        answer = "Sure, [SEG]."
        # Combine question + answer for teacher forcing
        input_text = question_formatted + answer


        #input_text = model_to_use.template['INSTRUCTION'].format(
            #input=text_for_res, round=1, bot_name=model_to_use.bot_name)

        input_ids = self.tokenizer.encode(input_text)
        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        # Forward pass WITH GRADIENTS
        data = {
            'pixel_values': [pixel_values],  # Pass as list for forward()
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': input_ids.clone(),  # Dummy labels
            'prompt_masks': None,
            'vp_overall_mask': None,
        }

        # Forward through model to get hidden states
        output = model_to_use(data)
        hidden_states = output.hidden_states[-1]  # Last layer hidden states

        # Apply text_hidden_fcs
        hidden_states = model_to_use.text_hidden_fcs(hidden_states)

        # Extract [SEG] token embedding
        seg_token_mask = input_ids == model_to_use.seg_token_idx

        if seg_token_mask.sum() == 0:
            # Fallback if no [SEG] token found
            if isinstance(gt_mask, torch.Tensor):
                h, w = gt_mask.shape[-2:]
            else:
                mask_array = np.array(gt_mask)
                h, w = mask_array.shape[:2]
            mask_binary = torch.zeros((h, w), dtype=torch.float32, device=self.device)
            mask_logits = torch.full((h, w), -10.0, dtype=torch.float32, device=self.device)
            return mask_binary, mask_logits

        pred_embeddings = hidden_states[seg_token_mask]  # (1, hidden_dim)
        pred_embeddings = pred_embeddings.squeeze(0)  # (hidden_dim,) - required for SAM2

        # Generate mask through SAM2 decoder (differentiable!)
        # Format embeddings as list of lists: [[embedding]] for inject_language_embd
        language_embeddings = [[pred_embeddings]]  # List[List[Tensor(hidden_dim,)]]
        sam_states = model_to_use.grounding_encoder.get_sam2_embeddings(g_pixel_values)
        pred_masks_list = model_to_use.grounding_encoder.inject_language_embd(
            sam_states, language_embeddings
        )

        pred_mask_raw = pred_masks_list[0]  # (1, H_sam, W_sam)

        # Resize to original image size
        w, h = ori_image_size
        pred_mask_resized = F.interpolate(
            pred_mask_raw.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)  # (H, W)

        # Binary mask for reward (detached)
        mask_binary = (pred_mask_resized.sigmoid() > 0.5).float().detach()

        # Return logits (before sigmoid) for loss - THIS HAS GRADIENTS!
        mask_logits = pred_mask_resized

        return mask_binary, mask_logits

    def _compute_dice_loss(self, pred_masks, gt_masks):
        """
        Compute Dice loss for each sample in the batch.

        Args:
            pred_masks: (B, H, W) tensor - predicted masks (after sigmoid)
            gt_masks: (B, H, W) tensor - ground truth masks

        Returns:
            dice_loss: (B,) tensor - Dice loss for each sample
        """
        smooth = 1e-5
        batch_size = pred_masks.shape[0]

        # Flatten spatial dimensions
        pred_flat = pred_masks.view(batch_size, -1)  # (B, H*W)
        gt_flat = gt_masks.view(batch_size, -1)  # (B, H*W)

        # Compute Dice coefficient
        intersection = (pred_flat * gt_flat).sum(dim=1)  # (B,)
        union = pred_flat.sum(dim=1) + gt_flat.sum(dim=1)  # (B,)

        dice = (2.0 * intersection + smooth) / (union + smooth)

        # Dice loss = 1 - Dice coefficient
        dice_loss = 1.0 - dice

        return dice_loss

    def _generate_mask_from_caption_trainable(self, image, caption, gt_mask):
        """
        Use training model to generate mask from caption (Referring Expression Segmentation).

        Note: For Loop 2, we use inference mode since the current loss implementation
        doesn't backprop through mask generation. Future: implement proper RL for mask generation.

        Args:
            image: PIL Image
            caption: str (referring expression)
            gt_mask: PIL Image or tensor (for getting correct size)

        Returns:
            mask: (H, W) tensor
            logprobs: tensor (dummy for now)
        """
        import numpy as np

        # Format text for RES (custom template for RL training)
        text_for_res = f'<image>\nThis is a description for something in the image: "{caption}", please segment it in the image.'

        # Use Sa2VA's predict_forward for RES inference
        # Note: Using eval mode for now - Loop 2 loss doesn't backprop through this yet
        with torch.no_grad():
            # Unwrap DDP if needed
            if hasattr(self.model, 'module'):
                model_to_use = self.model.module
            else:
                model_to_use = self.model

            out = model_to_use.predict_forward(
                image=image,
                text=text_for_res,
                tokenizer=self.tokenizer,
                return_logprobs=False
            )

        pred_masks = out.get('prediction_masks', None)

        if pred_masks is None or len(pred_masks) == 0:
            # Fallback: return empty mask with correct size
            if isinstance(gt_mask, torch.Tensor):
                h, w = gt_mask.shape[-2:]
            else:
                mask_array = np.array(gt_mask)
                h, w = mask_array.shape[:2]
            mask_tensor = torch.zeros((h, w), dtype=torch.float32, device=self.device)
        else:
            # Take first mask
            first_mask = pred_masks[0]

            # Convert to tensor
            if isinstance(first_mask, np.ndarray):
                mask_tensor = torch.from_numpy(first_mask).float()
            elif isinstance(first_mask, torch.Tensor):
                mask_tensor = first_mask.float()
            else:
                raise TypeError(f"Unexpected mask type: {type(first_mask)}")

            # Ensure 2D and move to correct device
            if mask_tensor.ndim > 2:
                mask_tensor = mask_tensor.squeeze()
            mask_tensor = mask_tensor.to(self.device)

        # Dummy logprobs for now (Loop 2 loss doesn't use it yet)
        logprobs = torch.tensor(0.0, device=self.device, requires_grad=True)

        return mask_tensor, logprobs

    def _clean_caption(self, caption):
        """
        Clean generated caption by removing special tokens and formatting.
        Uses the same cleaning logic as region captioning evaluation.

        Args:
            caption: raw caption string

        Returns:
            cleaned caption string
        """
        import re

        # Remove special tokens and formatting artifacts
        text_output = caption.replace("<s>", "").replace("\n", "") \
            .replace("region1", '').replace("Region1", '') \
            .replace("The region marked by", "").replace("The region marked as", "").replace("The region marked", "") \
            .replace(':', '') \
            .replace("   ", " ").replace("  ", " ")

        # Remove ASSISTANT: prefix if present
        text_output = text_output.split("ASSISTANT: ")[-1]

        # Remove HTML-like tags
        cleaned_str = re.sub(r'<.*?>', '', text_output)

        # Remove [SEG] tokens
        cleaned_str = cleaned_str.replace('[SEG]', '')

        # Clean up whitespace and quotes
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        return cleaned_str if cleaned_str else "object"

    def _generate_caption_from_mask_ema(self, image, mask):
        """
        Use EMA model to generate caption from mask (region captioning).

        Args:
            image: PIL Image
            mask: (H, W) tensor or numpy array

        Returns:
            caption: str
        """
        import re
        import numpy as np

        try:
            # Format text for region captioning (same format as evaluation)
            text_for_region_cap = "<image>\nPlease give me a detailed description of the region in the picture marked by region1."

            # Convert mask to numpy if needed
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy()
            else:
                mask_np = np.array(mask)

            # Ensure mask is uint8 and 2D
            if mask_np.ndim > 2:
                mask_np = mask_np.squeeze()
            mask_np = mask_np.astype(np.uint8)

            # Add batch dimension if needed: (H, W) -> (1, H, W)
            if mask_np.ndim == 2:
                mask_np = mask_np[np.newaxis, ...]

            # Use Sa2VA's predict_forward for region captioning
            with torch.no_grad():
                out = self.ema_model.predict_forward(
                    image=image,
                    text=text_for_region_cap,
                    tokenizer=self.tokenizer,
                    mask_prompts=[mask_np],  # Pass mask as mask_prompts
                    return_logprobs=False
                )

            prediction = out.get('prediction', '')

            # Clean up the prediction (same as region_cap eval)
            text_output = prediction.replace("<s>", "").replace("\n", "") \
                .replace("region1", '').replace("Region1", '').replace("The region marked by", "").replace("The region marked as", "").replace("The region marked", "") \
                .replace("is", "").replace("shows", "").replace(':', '').replace("   ", " ").replace("  ", " ")
            text_output = text_output.split("ASSISTANT: ")[-1]
            cleaned_str = re.sub(r'<.*?>', '', text_output)
            cleaned_str = cleaned_str.replace('[SEG]', '')
            cleaned_str = ' '.join(cleaned_str.split()).strip("'")
            cleaned_str = cleaned_str.strip()

            return cleaned_str if cleaned_str else "object"

        except Exception as e:
            # CRITICAL: Catch any exception to prevent training crash
            if self.rank == 0:
                print(f"\n{'='*70}")
                print(f"⚠ ERROR [Loop2]: EMA model failed to generate caption!")
                print(f"  Error: {type(e).__name__}: {str(e)}")
                print(f"  Returning default caption 'object' to continue training")
                print(f"{'='*70}\n")

            return "object"

    def _compute_meteor_rewards(self, pred_captions, gt_captions):
        """Compute METEOR rewards."""
        rewards = combined_caption_reward(
            gt_captions=gt_captions,
            pred_captions=pred_captions,
            llm_judge=None,
            use_llm_judge=False,
            meteor_weight=1.0,
            llm_judge_weight=0.0
        )
        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _compute_grpo_loss(self, prompt_ids, prompt_mask, completion_ids,
                          pixel_values, prompt_masks, vp_overall_mask,
                          rewards):
        """
        Compute GRPO loss.

        Args:
            prompt_ids: (batch, prompt_len) - includes padding
            completion_ids: (batch, completion_len) - generated tokens only (no prompt)

        This is a simplified version - full GRPO requires reference model log probs.
        For now, we use a policy gradient approach.
        """
        # Mask after EOS (handle cases with no EOS token)
        is_eos = completion_ids == self.tokenizer.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=self.device)
        has_eos = is_eos.any(dim=1)
        if has_eos.any():
            # Only compute argmax for sequences that have EOS token
            eos_positions = is_eos.int().argmax(dim=1)
            eos_idx[has_eos] = eos_positions[has_eos]
        sequence_indices = torch.arange(is_eos.size(1), device=self.device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate for forward pass
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        position_ids = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0).repeat(input_ids.size(0), 1)

        # Forward pass to get logits
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
        logits = outputs.logits[:, :-1, :]  # (B, L-1, V)
        target_ids = input_ids[:, 1:]  # (B, L-1)

        # Compute log probs for completion tokens with numerical stability
        # CRITICAL FIX: Clamp logits to prevent overflow in softmax
        logits = torch.clamp(logits, min=-100, max=100)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, dim=2, index=target_ids.unsqueeze(2)).squeeze(2)
        # CRITICAL FIX: Replace NaN log probs with large negative value
        token_log_probs = torch.nan_to_num(token_log_probs, nan=-100.0, posinf=0.0, neginf=-100.0)

        # Extract completion part
        # token_log_probs has shape (batch, seq_len-1) due to shift
        # input_ids = [prompt_ids | completion_ids]
        # We want log probs for completion part, which starts at position prompt_ids.shape[1]-1
        prompt_length = prompt_ids.shape[1]
        completion_log_probs = token_log_probs[:, prompt_length-1:]

        # Group-wise advantage normalization with numerical stability
        batch_size = len(rewards) // self.num_generations
        rewards_grouped = rewards.view(batch_size, self.num_generations)
        mean_rewards = rewards_grouped.mean(dim=1, keepdim=True)
        std_rewards = rewards_grouped.std(dim=1, keepdim=True)

        # CRITICAL FIX: Prevent NaN from division by zero when all rewards are identical
        # Use a larger epsilon and clamp std to prevent numerical instability
        epsilon = 1e-3
        std_rewards = torch.clamp(std_rewards, min=epsilon)

        advantages = (rewards_grouped - mean_rewards) / std_rewards
        advantages = advantages.view(-1)

        # CRITICAL FIX: Replace NaN/Inf advantages with zeros to prevent training collapse
        advantages = torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0)

        # Debug: Print advantages and completion log probs
        if hasattr(self, 'rank') and self.rank == 0:
            print(f"  [GRPO Debug] Rewards: {rewards.cpu().numpy()}")
            print(f"  [GRPO Debug] Mean rewards: {mean_rewards.squeeze().cpu().numpy()}")
            print(f"  [GRPO Debug] Std rewards: {std_rewards.squeeze().cpu().numpy()}")
            print(f"  [GRPO Debug] Advantages: {advantages.cpu().numpy()}")
            print(f"  [GRPO Debug] Completion log probs shape: {completion_log_probs.shape}")
            print(f"  [GRPO Debug] Completion log probs sum: {(completion_log_probs * completion_mask).sum(dim=1).float().detach().cpu().numpy()}")

        # Policy gradient loss
        per_token_loss = -completion_log_probs * advantages.unsqueeze(1)
        loss = (per_token_loss * completion_mask).sum() / (completion_mask.sum() + 1e-8)

        # CRITICAL FIX: Replace NaN loss with zero to prevent training collapse
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device, dtype=loss.dtype, requires_grad=True)

        # Debug: Print loss
        if hasattr(self, 'rank') and self.rank == 0:
            print(f"  [GRPO Debug] Loss: {loss.item():.6f}")

        metrics = {
            'reward': rewards.mean().item(),
            'reward_std': std_rewards.mean().item(),
            'completion_length': completion_mask.sum(1).float().mean().item(),
        }

        return loss, metrics

    def _add_image_tokens(self, texts):
        """Add image tokens to text."""
        IMG_TOKENS_PER_FRAME = 256
        image_token_str = f"<img>{'<IMG_CONTEXT>' * IMG_TOKENS_PER_FRAME}</img>"
        return [text.replace('<image>', image_token_str) for text in texts]

    def _stack_masks_with_resize(self, masks_list, device, target_size=448):
        """
        Stack masks with different sizes by resizing them to a unified size.

        Args:
            masks_list: List of masks with potentially different sizes [(H1, W1), (H2, W2), ...]
                       Can be tensors, numpy arrays, or PIL images
            device: torch device
            target_size: Target size for resizing (int or tuple). If int, resize to (target_size, target_size)

        Returns:
            Stacked tensor of shape (B, H, W) where H=W=target_size
        """
        if not masks_list:
            raise ValueError("masks_list is empty")

        # Determine target size
        if isinstance(target_size, int):
            target_h, target_w = target_size, target_size
        else:
            target_h, target_w = target_size

        resized_masks = []
        for mask in masks_list:
            # Convert PIL Image or numpy array to tensor
            if not isinstance(mask, torch.Tensor):
                if hasattr(mask, 'size'):  # PIL Image
                    mask = np.array(mask)
                mask = torch.tensor(mask, device=device)

            # Move to correct device
            mask = mask.to(device)

            # Handle different input shapes
            if mask.ndim == 2:
                # (H, W) -> (1, 1, H, W) for interpolate
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.ndim == 3:
                # (1, H, W) or (C, H, W) -> (1, C, H, W)
                mask = mask.unsqueeze(0)
            elif mask.ndim == 4:
                # Already (B, C, H, W) format
                pass
            else:
                raise ValueError(f"Unexpected mask shape: {mask.shape}")

            # Resize if needed
            if mask.shape[-2:] != (target_h, target_w):
                # Convert to float for interpolation
                mask_float = mask.float()
                mask_resized = F.interpolate(
                    mask_float,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                # Keep as float (preserve gradients if needed)
                mask = mask_resized

            # Remove extra dimensions: (1, 1, H, W) -> (H, W)
            mask = mask.squeeze(0).squeeze(0)
            resized_masks.append(mask)

        # Stack all masks
        stacked_masks = torch.stack(resized_masks, dim=0)  # (B, H, W)
        return stacked_masks

    def _repeat_masks(self, masks, num_repeats):
        """Repeat PIL masks for num_repeats times."""
        mask_tensors = []
        for mask in masks:
            mask_np = np.array(mask) > 0
            mask_tensor = torch.from_numpy(mask_np).float()
            mask_tensors.append(mask_tensor)

        # Use _stack_masks_with_resize to handle different sizes
        stacked = self._stack_masks_with_resize(mask_tensors, self.device)
        repeated = stacked.repeat_interleave(num_repeats, dim=0)
        return repeated

    def _update_ema_model(self):
        """Update EMA model parameters."""
        if self.ema_model is None:
            return  # Skip EMA update if not loaded on this rank
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

    def _compute_separate_gradients(self, loss_loop1, loss_loop2):
        """
        Compute gradients for Loop1 and Loop2 separately for analysis.

        Returns:
            (grad_loop1, grad_loop2): Tuple of flattened gradient tensors
        """
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # Compute Loop1 gradients
        self.model.zero_grad(set_to_none=True)
        loss_loop1.backward(retain_graph=True)
        grad_loop1_list = []
        for p in trainable_params:
            if p.grad is not None:
                grad_loop1_list.append(p.grad.flatten())
            else:
                grad_loop1_list.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        grad_loop1 = torch.cat(grad_loop1_list)

        # Compute Loop2 gradients
        self.model.zero_grad(set_to_none=True)
        loss_loop2.backward(retain_graph=True)
        grad_loop2_list = []
        for p in trainable_params:
            if p.grad is not None:
                grad_loop2_list.append(p.grad.flatten())
            else:
                grad_loop2_list.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
        grad_loop2 = torch.cat(grad_loop2_list)

        # ===== 额外：更有意义的 overlap cosine =====
        g1 = grad_loop1.float()
        g2 = grad_loop2.float()

        eps = 1e-12
        nonzero1 = g1.abs() > eps
        nonzero2 = g2.abs() > eps
        overlap = nonzero1 & nonzero2
        overlap_ratio = overlap.float().mean().item()

        if overlap.any():
            cos_overlap = F.cosine_similarity(g1[overlap].unsqueeze(0), g2[overlap].unsqueeze(0)).item()
        else:
            cos_overlap = 0.0


        return grad_loop1, grad_loop2, cos_overlap, overlap_ratio



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--max_steps', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--save_steps', type=int, default=250, help='Save checkpoint every N steps')
    parser.add_argument('--log_steps', type=int, default=10, help='Log detailed metrics every N steps')
    args = parser.parse_args()

    print("="*70)
    print("Sa2VA Dual-Loop RL Training - COMPLETE IMPLEMENTATION")
    print("="*70)
    import sys
    sys.stdout.flush()

    # Initialize distributed training
    import torch.distributed as dist
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # Running with torchrun - get rank BEFORE init so we can debug
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
            sys.stdout.flush()
            raise
    else:
        # Single GPU
        local_rank = 0
        world_size = 1
        device = "cuda"
        print("Single GPU training")
        sys.stdout.flush()

    # Initialize NLTK data AFTER distributed initialization
    # This ensures only rank 0 downloads, avoiding file lock conflicts
    print(f"\n[Rank {local_rank}] [1] Initializing NLTK data...")
    sys.stdout.flush()
    from projects.llava_sam2.rl_train.reward_functions import _ensure_nltk_data
    _ensure_nltk_data()
    print(f"[Rank {local_rank}] ✓ NLTK data ready")
    sys.stdout.flush()

    # Load dataset
    print(f"\n[Rank {local_rank}] [2] Loading dataset...")
    sys.stdout.flush()
    dataset = GraspAnyRegionDataset(local_data_dir=args.data_dir, parts_to_load=None)
    print(f"[Rank {local_rank}] ✓ Loaded {len(dataset)} samples")
    sys.stdout.flush()

    # Load model on CPU first to avoid OOM with multiple GPUs
    print(f"\n[Rank {local_rank}] [3] Loading training model...")
    sys.stdout.flush()

    # Load on CPU to save GPU memory during initialization
    model = Sa2VAChatModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    print(f"[Rank {local_rank}] Model loaded successfully!")
    sys.stdout.flush()

    # Apply LoRA before moving to GPU (more memory efficient)
    print(f"\n[Rank {local_rank}] [4] Applying LoRA...")
    sys.stdout.flush()
    model.wrap_llm_lora(r=128, lora_alpha=256, lora_dropout=0.05)

    # Now move to GPU
    model.to(device)

    # Freeze vision
    for param in model.vision_model.parameters():
        param.requires_grad = False
    print(f"[Rank {local_rank}] ✓ Vision frozen")
    sys.stdout.flush()

    # ===== Then unfreeze ONLY SAM2 decoder (mask_decoder) =====
    for p in model.grounding_encoder.parameters():
        p.requires_grad = False
    print(f"[Rank {local_rank}] ✓ grounding_encoder frozen")

    # Prepare for generation
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    # CRITICAL: Use same max_new_tokens as Loop 1 for consistency
    # Use 2048 (same as SFT training's default)
    MAX_COMPLETION_LENGTH = 2048
    model.preparing_for_generation(tokenizer, max_new_tokens=MAX_COMPLETION_LENGTH)

    # EMA model uses greedy decoding (do_sample=False) for deterministic inference
    # This is already the default from preparing_for_generation()

    # CRITICAL: Also update model.generation_config if it exists (from pretrained weights)
    # This prevents the model's default config from overriding our settings
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.max_new_tokens = MAX_COMPLETION_LENGTH
        print(f"[Rank {local_rank}] ✓ Overrode model.generation_config.max_new_tokens = {MAX_COMPLETION_LENGTH}")

    # Also override language_model's generation_config if it exists
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'generation_config'):
        if model.language_model.generation_config is not None:
            model.language_model.generation_config.max_new_tokens = MAX_COMPLETION_LENGTH
            print(f"[Rank {local_rank}] ✓ Overrode model.language_model.generation_config.max_new_tokens = {MAX_COMPLETION_LENGTH}")

    print(f"[Rank {local_rank}] ✓ Model prepared")
    print(f"  max_new_tokens: {MAX_COMPLETION_LENGTH}")
    print(f"  do_sample: False (greedy decoding for EMA inference)")
    sys.stdout.flush()

    # Wrap model in DDP for distributed training
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # find_unused_parameters=True needed because Loop 1/2 use different model parts
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False,broadcast_buffers=False)
        model._set_static_graph()
        # Enable static_graph to avoid "parameter marked ready twice" errors
        # This works because we're not computing separate gradients (compute_grads=False in training loop)
        #model._set_static_graph()
        print(f"[Rank {local_rank}] ✓ Model wrapped in DDP with static_graph=True")
        sys.stdout.flush()

    # Skip EMA model for now to save memory (use training model for both)
    # TODO: Re-enable EMA once we have proper memory management
    print("\n[5] Skipping EMA model (using training model for inference)...")
    if world_size > 1:
        ema_model = model.module  # Unwrap DDP for inference
    else:
        ema_model = model
    print("✓ EMA model = training model (temporary)")

    # Initialize preprocessor
    preprocessor = Sa2VADataPreprocessor()

    # Create trainer
    print("\n[6] Creating dual-loop trainer...")
    trainer = DualLoopRLTrainer(
        model=model,
        ema_model=ema_model,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        train_dataset=dataset,
        args=args,
        device=device,
        num_generations=args.num_generations,
    )
    trainer.rank = local_rank  # Set rank for multi-GPU training

    # Simple training loop
    print("\n[7] Starting training...")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate
    )

    from torch.utils.data import DataLoader, DistributedSampler

    # Use DistributedSampler for multi-GPU training
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

    # Calculate steps per epoch - directly from dataloader length
    steps_per_epoch = len(dataloader)

    if local_rank == 0:
        print(f"\n{'='*70}")
        print(f"Training Configuration:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  World size: {world_size}")
        print(f"  Batch size per GPU: {args.batch_size}")
        print(f"  Steps per epoch (batches): {steps_per_epoch}")
        print(f"  Number of epochs: {args.num_epochs}")
        print(f"  Save checkpoint every: {args.save_steps} steps")
        print(f"  Log detailed metrics every: {args.log_steps} steps")
        print(f"{'='*70}\n")

    global_step = 0

    # Tracking metrics for detailed logging
    iou_history = []
    meteor_history = []

    # Initialize TensorBoard writer (only on rank 0)
    writer = None
    if local_rank == 0:
        tensorboard_dir = os.path.join(args.output_dir, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
        print(f"\n✓ TensorBoard logging enabled: {tensorboard_dir}\n")

    for epoch in range(args.num_epochs):
        if local_rank == 0:
            print(f"\n{'='*70}")
            print(f"Starting Epoch {epoch + 1}/{args.num_epochs}")
            print(f"{'='*70}\n")

        # Set epoch for DistributedSampler to ensure proper shuffling
        if world_size > 1 and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        for step_in_epoch, batch in enumerate(dataloader):
            # Disable gradient analysis for DDP compatibility
            # The dual-loop backward() calls conflict with DDP's parameter tracking
            # TODO: Implement gradient analysis in a DDP-compatible way
            compute_grads = False
            #(world_size == 1) and (global_step % args.log_steps == 0)

            # Training step
            loss, metrics, separate_grads = trainer.train_step(batch, compute_separate_grads=compute_grads)


            # Track IoU and METEOR for detailed logging
            if 'loop1/mean_iou' in metrics:
                iou_history.append(metrics['loop1/mean_iou'])
            if 'loop2/mean_meteor' in metrics:
                meteor_history.append(metrics['loop2/mean_meteor'])

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Monitor gradient flow (Step 0 only)
            if global_step == 0 and local_rank == 0:
                print("\n" + "="*70)
                print("Gradient Flow Check (Step 0)")
                print("="*70)

                # Check LoRA parameters
                lora_params_with_grad = 0
                lora_params_total = 0
                for name, param in model.named_parameters():
                    if 'lora' in name.lower():
                        lora_params_total += 1
                        if param.grad is not None:
                            lora_params_with_grad += 1
                            grad_norm = param.grad.norm().item()
                            print(f"  ✓ LoRA param '{name}': grad_norm={grad_norm:.6f}")
                        else:
                            print(f"  ✗ LoRA param '{name}': NO GRADIENT")

                print(f"\nLoRA gradient summary: {lora_params_with_grad}/{lora_params_total} params have gradients")

                # Check SAM2 decoder parameters
                sam2_decoder_params_with_grad = 0
                sam2_decoder_params_total = 0
                for name, param in model.named_parameters():
                    if 'grounding_encoder' in name and 'mask_decoder' in name:
                        sam2_decoder_params_total += 1
                        if param.grad is not None:
                            sam2_decoder_params_with_grad += 1
                            grad_norm = param.grad.norm().item()
                            print(f"  ✓ SAM2 decoder param '{name}': grad_norm={grad_norm:.6f}")
                        else:
                            print(f"  ✗ SAM2 decoder param '{name}': NO GRADIENT")

                print(f"\nSAM2 decoder gradient summary: {sam2_decoder_params_with_grad}/{sam2_decoder_params_total} params have gradients")
                print("="*70 + "\n")

            optimizer.step()

            # Clear CUDA cache every step to prevent OOM from fragmentation
            if global_step % 1 == 0:
                torch.cuda.empty_cache()

            # Regular logging (every step)
            if global_step % 1 == 0:
                print(f"[Rank {local_rank}] Epoch {epoch+1} Step {step_in_epoch}/{steps_per_epoch}: "
                      f"loss={loss.item():.4f}, "
                      f"loop1_loss={metrics['loop1_loss']:.4f}, "
                      f"loop2_loss={metrics['loop2_loss']:.4f}")

            # TensorBoard logging: EVERY STEP for detailed monitoring
            if local_rank == 0 and writer is not None:
                # Log scalar metrics
                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/loop1', metrics['loop1_loss'], global_step)
                writer.add_scalar('Loss/loop2', metrics['loop2_loss'], global_step)

                # Log Loop1 metrics
                if 'loop1/mean_iou' in metrics:
                    writer.add_scalar('Loop1/mean_iou', metrics['loop1/mean_iou'], global_step)
                if 'loop1/reward' in metrics:
                    writer.add_scalar('Loop1/reward', metrics['loop1/reward'], global_step)
                if 'loop1/completion_length' in metrics:
                    writer.add_scalar('Loop1/completion_length', metrics['loop1/completion_length'], global_step)

                # Log Loop2 metrics
                if 'loop2/mean_meteor' in metrics:
                    writer.add_scalar('Loop2/mean_meteor', metrics['loop2/mean_meteor'], global_step)
                if 'loop2/reward' in metrics:
                    writer.add_scalar('Loop2/reward', metrics['loop2/reward'], global_step)
                if 'loop2/bce_loss' in metrics:
                    writer.add_scalar('Loop2/bce_loss', metrics['loop2/bce_loss'], global_step)
                if 'loop2/dice_loss' in metrics:
                    writer.add_scalar('Loop2/dice_loss', metrics['loop2/dice_loss'], global_step)

                # Log gradient analysis (only when available)
                if separate_grads is not None:
                    grad_loop1, grad_loop2, cos_overlap, overlap_ratio = separate_grads
                    cosine_all = F.cosine_similarity(grad_loop1.float().unsqueeze(0), grad_loop2.float().unsqueeze(0)).item()
                    writer.add_scalar('Gradients/cosine_all', cosine_all, global_step)
                    writer.add_scalar('Gradients/cosine_overlap', cos_overlap, global_step)
                    writer.add_scalar('Gradients/overlap_ratio', overlap_ratio, global_step)


                # Flush to ensure data is written immediately
                writer.flush()

            # Detailed logging every log_steps
            if local_rank == 0 and global_step > 0 and global_step % args.log_steps == 0:
                print(f"\n{'='*70}")
                print(f"DETAILED METRICS @ Step {global_step}")
                print(f"{'='*70}")

                # IoU statistics (last log_steps batches)
                if len(iou_history) > 0:
                    recent_ious = iou_history[-args.log_steps:]
                    print(f"IoU (Loop1) - Last {len(recent_ious)} batches:")
                    print(f"  Min:  {min(recent_ious):.4f}")
                    print(f"  Max:  {max(recent_ious):.4f}")
                    print(f"  Mean: {sum(recent_ious)/len(recent_ious):.4f}")

                # METEOR statistics
                if len(meteor_history) > 0:
                    recent_meteors = meteor_history[-args.log_steps:]
                    print(f"METEOR (Loop2) - Last {len(recent_meteors)} batches:")
                    print(f"  Min:  {min(recent_meteors):.4f}")
                    print(f"  Max:  {max(recent_meteors):.4f}")
                    print(f"  Mean: {sum(recent_meteors)/len(recent_meteors):.4f}")

                # Gradient cosine similarity
                if separate_grads is not None:
                    grad_loop1, grad_loop2, cos_overlap, overlap_ratio = separate_grads
                    cosine_all = F.cosine_similarity(grad_loop1.float().unsqueeze(0),grad_loop2.float().unsqueeze(0)).item()

                    #print(f"Gradient Analysis:")
                    #print(f"  Loop1 grad norm: {grad_loop1.norm().item():.6f}")
                    #print(f"  Loop2 grad norm: {grad_loop2.norm().item():.6f}")
                    #print(f"  Cosine similarity: {cosine_sim.item():.6f}")

                print(f"{'='*70}\n")

            # Save checkpoint every save_steps
            if local_rank == 0 and global_step > 0 and global_step % args.save_steps == 0:
                print(f"\n{'='*70}")
                print(f"SAVING CHECKPOINT @ Step {global_step}")
                print(f"{'='*70}")
                model_to_save = model.module if hasattr(model, 'module') else model
                checkpoint_dir = f"{args.output_dir}/checkpoint-{global_step}"
                model_to_save.save_pretrained(checkpoint_dir)
                print(f"✓ Checkpoint saved to {checkpoint_dir}")
                print(f"{'='*70}\n")

            global_step += 1

        if local_rank == 0:
            print(f"\n{'='*70}")
            print(f"✓ Epoch {epoch + 1}/{args.num_epochs} completed ({step_in_epoch + 1} steps)")
            print(f"{'='*70}\n")

    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)

    # Close TensorBoard writer
    if local_rank == 0 and writer is not None:
        writer.close()
        print("✓ TensorBoard writer closed")

    # Save model (only on rank 0)
    if local_rank == 0:
        # Unwrap DDP if needed
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(f"{args.output_dir}/final_model")
        print(f"✓ Model saved to {args.output_dir}/final_model")


if __name__ == "__main__":
    main()
