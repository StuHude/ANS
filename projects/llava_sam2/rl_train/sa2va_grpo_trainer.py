"""
Sa2VA GRPO Trainer - Adapted from R1-V's Qwen2VLGRPOTrainer

This trainer adapts the GRPO (Group Relative Policy Optimization) algorithm
for Sa2VA model, which handles both image understanding and mask generation.

Key adaptations from R1-V:
1. Sa2VA-specific model loading and preprocessing
2. Dual-loop training support (mask→caption and caption→mask)
3. Integration with Sa2VA's data format (pixel_values, prompt_masks, etc.)
4. EMA model support for model 2

Reference: /data/xiaoyicheng/Sa2VA/R1-V/src/r1-v/src/open_r1/trainer/grpo_trainer.py
"""

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import copy

import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    GenerationConfig,
    Trainer,
    TrainerCallback,
    is_wandb_available,
    LogitsProcessorList,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# Import Sa2VA components
from projects.llava_sam2.rl_train.data_preprocessor import Sa2VADataPreprocessor
from projects.llava_sam2.rl_train.tokenization import Sa2VATemplateAndTokenizer
from projects.llava_sam2.rl_train.logits_processor import (
    NumericalStabilityLogitsProcessor,
    TemperatureLogitsWarper,
)

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards.
RewardFunc = Callable[[list, list], list[float]]


class Sa2VAGRPOTrainer(Trainer):
    """
    GRPO Trainer adapted for Sa2VA model (based on R1-V's Qwen2VLGRPOTrainer).

    Differences from R1-V's Qwen2VLGRPOTrainer:
    1. Uses Sa2VA model instead of Qwen2-VL
    2. Uses Sa2VADataPreprocessor for input preparation
    3. Supports dual-loop training with different reward functions
    4. Handles Sa2VA's unique input format (pixel_values, prompt_masks, etc.)

    Args:
        model: Sa2VA model (already loaded and LoRA-wrapped)
        reward_funcs: List of reward functions
        args: GRPOConfig
        train_dataset: Dataset with samples containing image, mask, caption
        eval_dataset: Optional evaluation dataset
        processing_class: Tokenizer for Sa2VA
        callbacks: Optional callbacks
        optimizers: Optional optimizer/scheduler tuple
        peft_config: PEFT config (not used, Sa2VA's LoRA is already applied)
        preprocessor: Sa2VADataPreprocessor instance
        template_tokenizer: Sa2VATemplateAndTokenizer instance
    """

    def __init__(
        self,
        model: PreTrainedModel,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        preprocessor: Optional[Sa2VADataPreprocessor] = None,
        template_tokenizer: Optional[Sa2VATemplateAndTokenizer] = None,
    ):
        # Args
        if args is None:
            model_name = model.config._name_or_path.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Model is already loaded and LoRA-wrapped externally
        # Reference model
        if is_deepspeed_zero3_enabled():
            # For DeepSpeed, load a fresh reference model
            from projects.llava_sam2.hf.models.modeling_sa2va_chat import Sa2VAChatModel
            model_path = model.config._name_or_path
            model_init_kwargs = args.model_init_kwargs or {}
            self.ref_model = Sa2VAChatModel.from_pretrained(model_path, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used (Sa2VA has LoRA), reference model is not needed since adapter can be disabled
            self.ref_model = None

        # Processing class (tokenizer)
        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")

        pad_token_id = processing_class.pad_token_id
        if pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
            pad_token_id = processing_class.pad_token_id

        # Preprocessor for Sa2VA
        if preprocessor is None:
            preprocessor = Sa2VADataPreprocessor()
        self.preprocessor = preprocessor

        # Template tokenizer
        if template_tokenizer is None:
            template_tokenizer = Sa2VATemplateAndTokenizer(processing_class, max_length=8196)
        self.template_tokenizer = template_tokenizer

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        self.reward_funcs = reward_funcs

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # CRITICAL: Setup logits processors for numerical stability
        # This prevents NaN/inf issues during generation
        self.logits_processor = LogitsProcessorList([
            NumericalStabilityLogitsProcessor(clip_value=30.0, min_prob=1e-8, verbose=True),
            # Temperature is already set in generation_config, but we add extra safety
            TemperatureLogitsWarper(temperature=1.0, min_temperature=0.1),
        ])
        print("✓ Logits processors configured for numerical stability")

        # Suppress the "Could not estimate the number of tokens" warning
        model.warnings_issued = getattr(model, "warnings_issued", {})
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # Set signature columns to what our dataset provides
        if self._signature_columns is None:
            self._signature_columns = ["image", "mask", "caption", "category", "image_id"]

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, prompt_masks, vp_overall_mask):
        """
        Compute per-token log probabilities for Sa2VA model.

        Args:
            model: Sa2VA model (may be wrapped in DataParallel)
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            pixel_values: (B, 1, 3, 448, 448) or (B, 3, 448, 448) image tensors
            prompt_masks: List of (N_regions, G, G) tensors, length B
            vp_overall_mask: (B,) overall mask indicators

        Returns:
            per_token_logps: (B, L-1) log probabilities for each token
        """
        # CRITICAL FIX: Unwrap DataParallel if needed
        # DataParallel wraps the model and changes the forward signature,
        # but Sa2VA expects forward(data, mode) not forward(*args, **kwargs)
        from torch.nn.parallel import DataParallel
        if isinstance(model, DataParallel):
            sa2va_model = model.module
        else:
            sa2va_model = model

        # Sa2VA forward pass - pack parameters into data dict as expected by Sa2VA model
        bs, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).repeat(bs, 1)

        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': input_ids.clone(),  # For GRPO, we use input_ids as labels to compute log probs
            'pixel_values': pixel_values,
            'prompt_masks': prompt_masks,
            'vp_overall_mask': vp_overall_mask,
        }
        outputs = sa2va_model(data, mode='loss')
        logits = outputs.logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID

        # Compute the log probabilities for the input tokens
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        # We preprocess the data in `compute_loss`, so skip the default preparation
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Main training loop for Sa2VA GRPO.

        Flow:
        1. Preprocess raw data (image, mask, caption) -> Sa2VA format
        2. Generate completions (captions) for each prompt
        3. Compute rewards using reward functions
        4. Compute GRPO loss with group-wise normalization
        """
        if return_outputs:
            raise ValueError("The Sa2VAGRPOTrainer does not support returning outputs")

        # Extract raw data from inputs
        # Each input should contain: image, mask, caption, etc.
        images = [x["image"] for x in inputs]
        masks = [x["mask"] for x in inputs]
        gt_captions = [x["caption"] for x in inputs]

        batch_size = len(images)
        device = self.accelerator.device

        # ========================================================================
        # Step 1: Preprocess data to Sa2VA format
        # ========================================================================

        # For mask->caption task: we have image + mask, need to generate caption
        # Preprocess each sample
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
        pixel_values = torch.stack([s['pixel_values'] for s in preprocessed_samples]).to(device)  # (B, 1, 3, 448, 448)
        # CRITICAL: prompt_masks must be a LIST, not a stacked tensor!
        # Each element is (N_regions, G, G), for single region: (1, 16, 16)
        prompt_masks = [s['prompt_masks'].to(device) for s in preprocessed_samples]  # List of (1, 16, 16) tensors
        vp_overall_mask = torch.stack([s['vp_overall_mask'] for s in preprocessed_samples]).to(device).squeeze(-1)  # (B,)

        # Get prompt texts (for generation, not tokenized yet)
        prompt_texts = [s['prompt_text'] for s in preprocessed_samples]

        # CRITICAL FIX: Replace <image> with <img><IMG_CONTEXT>*256</img> BEFORE tokenization
        # This matches the SFT training pipeline in describe_anything_referring_dataset.py
        IMG_TOKENS_PER_FRAME = 256  # From data_preprocessor.py: GRID_SIZE * GRID_SIZE = 16 * 16
        num_image_tokens = 1 * IMG_TOKENS_PER_FRAME  # 1 tile per sample
        image_token_str = f"<img>{'<IMG_CONTEXT>' * num_image_tokens}</img>"
        prompt_texts = [text.replace('<image>', image_token_str) for text in prompt_texts]

        # Tokenize prompts
        prompt_encodings = self.processing_class(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True,
        )
        prompt_ids = prompt_encodings["input_ids"].to(device)  # (B, P)
        prompt_mask = prompt_encodings["attention_mask"].to(device)  # (B, P)

        # Truncate prompts if needed
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ========================================================================
        # Step 2: Generate completions
        # ========================================================================

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # Generate G completions per prompt
            # Note: Sa2VA's generate method needs pixel_values, prompt_masks, etc.
            # CRITICAL: Pass logits_processor to prevent NaN/inf during generation
            prompt_completion_ids = unwrapped_model.generate(
                input_ids=prompt_ids,
                pixel_values=pixel_values,
                prompt_masks=prompt_masks,
                vp_overall_mask=vp_overall_mask,
                generation_config=self.generation_config,
                attention_mask=prompt_mask,
                logits_processor=self.logits_processor,  # Add numerical stability
            )

            # Split into prompt and completion
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Repeat inputs for num_generations
            prompt_ids = prompt_ids.repeat_interleave(self.num_generations, dim=0)
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
            pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0)
            # CRITICAL: prompt_masks is a list, need to repeat each element
            prompt_masks = [mask for mask in prompt_masks for _ in range(self.num_generations)]
            vp_overall_mask = vp_overall_mask.repeat_interleave(self.num_generations, dim=0)

        # ========================================================================
        # Step 3: Mask everything after the first EOS token
        # ========================================================================

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt and completion for full sequence
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        # ========================================================================
        # Step 4: Compute log probabilities
        # ========================================================================

        per_token_logps = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask,
            pixel_values, prompt_masks, vp_overall_mask
        )
        # Get rid of the prompt (-1 because of the shift done in _get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        # Compute reference model log probabilities
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask,
                    pixel_values, prompt_masks, vp_overall_mask
                )
            else:
                # Use adapter disabling for PEFT
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask,
                        pixel_values, prompt_masks, vp_overall_mask
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # ========================================================================
        # Step 5: Decode completions
        # ========================================================================

        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)

        # ========================================================================
        # Step 6: Compute rewards
        # ========================================================================

        # Prepare data for reward functions
        # Reward functions expect (prompts, completions, **kwargs)
        # We need to repeat gt_captions for each generation
        prompts = [prompt_texts[i // self.num_generations] for i in range(batch_size * self.num_generations)]
        gt_captions_repeated = [gt_captions[i // self.num_generations] for i in range(batch_size * self.num_generations)]

        # Compute rewards from all reward functions
        rewards_per_func = torch.zeros(batch_size * self.num_generations, len(self.reward_funcs), device=device)

        for i, reward_func in enumerate(self.reward_funcs):
            # Call reward function with prompts, completions, and gt_captions
            # Also pass any additional kwargs needed (llm_judge, etc.)
            reward_kwargs = {
                'gt_captions': gt_captions_repeated,
            }

            # Add any extra columns from inputs (excluding standard ones)
            for key in inputs[0].keys():
                if key not in ["image", "mask", "caption"]:
                    reward_kwargs[key] = [x[key] for x in inputs for _ in range(self.num_generations)]

            output_rewards = reward_func(
                prompts=prompts,
                completions=completions,
                **reward_kwargs
            )
            rewards_per_func[:, i] = torch.tensor(output_rewards, dtype=torch.float32, device=device)

        # Sum rewards from all functions
        rewards = rewards_per_func.sum(dim=1)

        # ========================================================================
        # Step 7: Compute grouped-wise rewards and advantages
        # ========================================================================

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # ========================================================================
        # Step 8: Compute GRPO loss
        # ========================================================================

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # ========================================================================
        # Step 9: Log metrics
        # ========================================================================

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """Creates a draft of a model card using the information available to the Trainer."""
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
