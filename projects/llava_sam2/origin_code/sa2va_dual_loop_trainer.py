"""
Sa2VA Dual-Loop GRPO Trainer

This trainer implements simultaneous training of two loops:
- Loop 1: mask → caption (reward on caption quality)
- Loop 2: caption → mask → caption' (reward on caption' quality + mask quality)

Both loops are executed for each batch and their losses are combined.
"""

import torch
import torch.nn.functional as F
from typing import Any, Union

from projects.llava_sam2.rl_train.sa2va_grpo_trainer import Sa2VAGRPOTrainer
from trl.models import unwrap_model_for_generation


class Sa2VADualLoopGRPOTrainer(Sa2VAGRPOTrainer):
    """
    GRPO Trainer with dual-loop training for Sa2VA.

    Each training step executes both:
    1. Loop 1 (mask→caption): Generate caption from mask, compute reward
    2. Loop 2 (caption→mask→caption'): Generate mask from caption, then caption' from mask

    The losses from both loops are combined.

    Args:
        loop1_weight: Weight for loop 1 loss (default: 0.5)
        loop2_weight: Weight for loop 2 loss (default: 0.5)
        **kwargs: Arguments for Sa2VAGRPOTrainer
    """

    def __init__(self, loop1_weight=0.5, loop2_weight=0.5, **kwargs):
        super().__init__(**kwargs)
        self.loop1_weight = loop1_weight
        self.loop2_weight = loop2_weight

        # Verify we have exactly 2 reward functions (one for each loop)
        if len(self.reward_funcs) != 2:
            raise ValueError(f"Dual-loop trainer requires exactly 2 reward functions, got {len(self.reward_funcs)}")

        self.reward_func_loop1 = self.reward_funcs[0]  # mask→caption reward
        self.reward_func_loop2 = self.reward_funcs[1]  # caption→mask→caption' reward

        print(f"✓ Dual-loop trainer initialized:")
        print(f"  Loop 1 weight: {self.loop1_weight}")
        print(f"  Loop 2 weight: {self.loop2_weight}")
        print(f"  Reward func loop 1: {self.reward_func_loop1.__name__}")
        print(f"  Reward func loop 2: {self.reward_func_loop2.__name__}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Dual-loop training compute_loss.

        For each batch:
        1. Execute Loop 1: mask → caption
        2. Execute Loop 2: caption → mask → caption'
        3. Combine losses
        """
        if return_outputs:
            raise ValueError("The Sa2VADualLoopGRPOTrainer does not support returning outputs")

        # Extract raw data
        images = [x["image"] for x in inputs]
        masks = [x["mask"] for x in inputs]
        gt_captions = [x["caption"] for x in inputs]
        batch_size = len(images)
        device = self.accelerator.device

        # =====================================================================
        # LOOP 1: mask → caption
        # =====================================================================

        loss_loop1, metrics_loop1 = self._compute_loop1_loss(
            model, images, masks, gt_captions, device
        )

        # =====================================================================
        # LOOP 2: caption → mask → caption'
        # =====================================================================

        loss_loop2, metrics_loop2 = self._compute_loop2_loss(
            model, images, masks, gt_captions, device
        )

        # =====================================================================
        # Combine losses
        # =====================================================================

        total_loss = self.loop1_weight * loss_loop1 + self.loop2_weight * loss_loop2

        # Log metrics with loop prefixes
        for key, value in metrics_loop1.items():
            self._metrics[f"loop1/{key}"].append(value)

        for key, value in metrics_loop2.items():
            self._metrics[f"loop2/{key}"].append(value)

        self._metrics["loop1_loss"].append(loss_loop1.item())
        self._metrics["loop2_loss"].append(loss_loop2.item())
        self._metrics["total_loss"].append(total_loss.item())

        return total_loss

    def _compute_loop1_loss(self, model, images, masks, gt_captions, device):
        """
        Loop 1: mask → caption

        Given: image + mask
        Generate: caption
        Reward: caption quality (METEOR + LLM judge)
        """
        batch_size = len(images)

        # Preprocess for mask→caption task
        preprocessed_samples = []
        for image, mask, caption in zip(images, masks, gt_captions):
            preprocessed = self.preprocessor.prepare_for_model(
                image=image,
                mask=mask,
                caption=caption,
                task="mask_to_caption",
                instruction="Please generate a detailed description for the given image region."
            )
            preprocessed_samples.append(preprocessed)

        # Stack tensors
        pixel_values = torch.stack([s['pixel_values'] for s in preprocessed_samples]).to(device)
        prompt_masks = [s['prompt_masks'].to(device) for s in preprocessed_samples]
        vp_overall_mask = torch.stack([s['vp_overall_mask'] for s in preprocessed_samples]).to(device).squeeze(-1)

        # Get prompt texts and tokenize
        prompt_texts = [s['prompt_text'] for s in preprocessed_samples]
        IMG_TOKENS_PER_FRAME = 256
        num_image_tokens = 1 * IMG_TOKENS_PER_FRAME
        image_token_str = f"<img>{'<IMG_CONTEXT>' * num_image_tokens}</img>"
        prompt_texts = [text.replace('<image>', image_token_str) for text in prompt_texts]

        prompt_encodings = self.processing_class(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )
        prompt_ids = prompt_encodings["input_ids"].to(device)
        prompt_mask = prompt_encodings["attention_mask"].to(device)

        # Truncate if needed
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                prompt_masks=prompt_masks,
                vp_overall_mask=vp_overall_mask,
                generation_config=self.generation_config,
                attention_mask=prompt_mask,
                logits_processor=self.logits_processor,
            )

            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Repeat inputs for num_generations
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0)
            prompt_masks = [mask for mask in prompt_masks for _ in range(self.num_generations)]
            vp_overall_mask = vp_overall_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask after EOS
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # Compute log probabilities
        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask,
            pixel_values, prompt_masks, vp_overall_mask
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        # Reference model log probs
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    pixel_values, prompt_masks, vp_overall_mask
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask,
                        pixel_values, prompt_masks, vp_overall_mask
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # Compute rewards (Loop 1)
        prompts = [prompt_texts[i // self.num_generations] for i in range(batch_size * self.num_generations)]
        gt_captions_repeated = [gt_captions[i // self.num_generations] for i in range(batch_size * self.num_generations)]

        reward_kwargs = {'gt_captions': gt_captions_repeated}
        output_rewards = self.reward_func_loop1(prompts=prompts, completions=completions, **reward_kwargs)
        rewards = torch.tensor(output_rewards, dtype=torch.float32, device=device)

        # Group-wise normalization
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # GRPO loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # Metrics
        metrics = {
            'reward': rewards.mean().item(),
            'reward_std': std_grouped_rewards.mean().item(),
            'kl': ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean().item(),
            'completion_length': completion_mask.sum(1).float().mean().item(),
        }

        return loss, metrics

    def _compute_loop2_loss(self, model, images, masks, gt_captions, device):
        """
        Loop 2: caption → mask → caption'

        Step 1: Given image + GT caption, generate mask
        Step 2: Given image + generated mask, generate caption'
        Reward: caption' quality + mask quality (IoU)

        NOTE: This is a simplified version that only does caption→caption
        (skipping mask generation for now, which requires Sa2VA to support mask output)

        For now, we'll use the same flow as Loop 1 but with different prompting.
        """
        # PLACEHOLDER: Currently same as Loop 1
        # TODO: Implement actual caption→mask→caption' flow when Sa2VA supports mask generation

        # For now, return zero loss to avoid errors
        # This should be properly implemented with mask generation support
        print("⚠ WARNING: Loop 2 (caption→mask→caption') not fully implemented yet")
        print("  Currently using placeholder implementation")

        loss = torch.tensor(0.0, device=device, requires_grad=True)
        metrics = {
            'reward': 0.0,
            'reward_std': 0.0,
            'kl': 0.0,
            'completion_length': 0.0,
        }

        return loss, metrics
