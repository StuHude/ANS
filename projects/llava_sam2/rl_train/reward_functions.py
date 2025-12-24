"""
Reward functions for Sa2VA RL training:
1. IOU reward: for mask prediction accuracy
2. METEOR reward: for caption generation quality
3. LLM judge reward: for caption quality evaluation using OpenAI API

Reference: repos/describe-anything/evaluation/eval_model_outputs.py
"""

import torch
import numpy as np
from typing import List, Optional
from nltk.translate.meteor_score import meteor_score
import nltk
from openai import OpenAI
import os

# Download METEOR data if needed
# Fix for DDP: Only rank 0 downloads to avoid race condition
import torch.distributed as dist

def _ensure_nltk_data():
    """Check NLTK data availability (assumes already downloaded)."""
    import os
    import time
    if dist.is_available() and dist.is_initialized():
        # Distributed training: only rank 0 checks
        rank = dist.get_rank()
        if rank == 0:
            # Just verify data exists, don't download
            print("[Rank 0] Checking NLTK data...")
            try:
                nltk.data.find('corpora/wordnet')
                print("[Rank 0] ✓ NLTK wordnet found")
            except LookupError:
                print("[Rank 0] ⚠ NLTK wordnet not found (expected at /root/nltk_data/corpora/wordnet)")

            try:
                nltk.data.find('corpora/omw-1.4')
                print("[Rank 0] ✓ NLTK omw-1.4 found")
            except LookupError:
                print("[Rank 0] ⚠ NLTK omw-1.4 not found (expected at /root/nltk_data/corpora/omw-1.4)")

        # Barrier: all ranks wait for rank 0 to finish checking
        print(f"[Rank {rank}] Waiting at barrier for NLTK data...")
        dist.barrier()
        print(f"[Rank {rank}] ✓ Passed barrier, NLTK data ready")
    else:
        # Single GPU: just verify
        try:
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/omw-1.4')
        except LookupError as e:
            print(f"⚠ NLTK data missing: {e}")

# DO NOT call _ensure_nltk_data() at module import time!
# It will be called in main() after distributed initialization
# _ensure_nltk_data()  # COMMENTED OUT - will be called explicitly in training script


def compute_iou(mask1, mask2):
    """
    Compute Intersection over Union (IoU) between two masks.

    Args:
        mask1: numpy array or torch.Tensor (H, W) or (1, H, W)
        mask2: numpy array or torch.Tensor (H, W) or (1, H, W)

    Returns:
        float: IoU score between 0 and 1
    """
    # Convert to numpy if needed
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()

    # Squeeze if needed
    while mask1.ndim > 2:
        mask1 = mask1.squeeze(0)
    while mask2.ndim > 2:
        mask2 = mask2.squeeze(0)

    # Ensure boolean
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Compute intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return float(iou)


def compute_meteor(reference, hypothesis):
    """
    Compute METEOR score between reference and hypothesis captions.

    Args:
        reference: str, ground truth caption
        hypothesis: str, predicted caption

    Returns:
        float: METEOR score between 0 and 1
    """
    # Tokenize
    ref_tokens = reference.strip().lower().split()
    hyp_tokens = hypothesis.strip().lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    try:
        score = meteor_score([ref_tokens], hyp_tokens)
    except Exception as e:
        print(f"METEOR computation error: {e}")
        score = 0.0

    return score


def iou_reward_batch(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]) -> List[float]:
    """
    Compute IOU rewards for a batch of mask pairs.

    Args:
        gt_masks: List of ground truth masks
        pred_masks: List of predicted masks

    Returns:
        List of IOU scores
    """
    assert len(gt_masks) == len(pred_masks), "Batch sizes must match"

    rewards = []
    for gt_mask, pred_mask in zip(gt_masks, pred_masks):
        iou = compute_iou(gt_mask, pred_mask)
        rewards.append(iou)

    return rewards


def meteor_reward_batch(gt_captions: List[str], pred_captions: List[str]) -> List[float]:
    """
    Compute METEOR rewards for a batch of caption pairs.

    Args:
        gt_captions: List of ground truth captions
        pred_captions: List of predicted captions

    Returns:
        List of METEOR scores
    """
    assert len(gt_captions) == len(pred_captions), "Batch sizes must match"

    rewards = []
    for gt_cap, pred_cap in zip(gt_captions, pred_captions):
        meteor = compute_meteor(gt_cap, pred_cap)
        rewards.append(meteor)

    return rewards


# ============================================================================
# LLM Judge Reward (following describe-anything evaluation)
# ============================================================================

class LLMJudge:
    """
    LLM-based caption quality judge using OpenAI API.
    Simplified version of describe-anything evaluation for RL training.
    """

    # Prompt for LLM judge
    JUDGE_PROMPT = """You are an expert evaluator of image descriptions. Your task is to rate the quality of a generated description for an image region.

Reference description (ground truth):
{reference}

Generated description (to evaluate):
{hypothesis}

Please rate the generated description on a scale from 0 to 10 based on these criteria:
1. **Accuracy**: Does it correctly describe the content?
2. **Completeness**: Does it capture important details from the reference?
3. **Clarity**: Is it clear and easy to understand?
4. **Relevance**: Does it focus on relevant information?

IMPORTANT:
- The descriptions don't need to match word-for-word
- Focus on semantic similarity and content accuracy
- Minor wording differences are acceptable if the meaning is preserved
- Missing minor details should result in small deductions only

Please output ONLY a single number between 0 and 10 (can include decimals like 7.5).
Do NOT include any explanation, reasoning, or other text. Just the number.

Rating:"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:9100/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        temperature: float = 0.0,
        max_tokens: int = 10
    ):
        """
        Initialize LLM judge.

        Args:
            api_key: OpenAI API key (or None for local server)
            base_url: Base URL for API (default: local vLLM server)
            model: Model name/ID
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Max tokens for response
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "api_key")
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def judge_single(self, reference: str, hypothesis: str) -> float:
        """
        Judge a single caption pair.

        Args:
            reference: Ground truth caption
            hypothesis: Generated caption

        Returns:
            Score between 0 and 1
        """
        prompt = self.JUDGE_PROMPT.format(
            reference=reference,
            hypothesis=hypothesis
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            # Extract score from response
            message = response.choices[0].message.content.strip()

            # Parse the score
            try:
                score = float(message)
                # Normalize to [0, 1]
                score = max(0.0, min(10.0, score)) / 10.0
            except ValueError:
                # If parsing fails, try to extract first number
                import re
                numbers = re.findall(r'\d+\.?\d*', message)
                if numbers:
                    score = float(numbers[0]) / 10.0
                else:
                    print(f"Warning: Could not parse LLM judge response: {message}")
                    score = 0.0

            return score

        except Exception as e:
            print(f"Error in LLM judge: {e}")
            return 0.0

    def judge_batch(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """
        Judge a batch of caption pairs.

        Args:
            references: List of ground truth captions
            hypotheses: List of generated captions

        Returns:
            List of scores between 0 and 1
        """
        assert len(references) == len(hypotheses), "Batch sizes must match"

        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.judge_single(ref, hyp)
            scores.append(score)

        return scores


def combined_caption_reward(
    gt_captions: List[str],
    pred_captions: List[str],
    llm_judge: Optional[LLMJudge] = None,
    use_llm_judge: bool = False,
    meteor_weight: float = 0.25,
    llm_judge_weight: float = 0.75
) -> List[float]:
    """
    Compute combined caption reward:
    - If use_llm_judge=False: reward = METEOR (100%)
    - If use_llm_judge=True: reward = meteor_weight * METEOR + llm_judge_weight * LLM_judge_score

    Args:
        gt_captions: List of ground truth captions
        pred_captions: List of predicted captions
        llm_judge: LLMJudge instance (required if use_llm_judge=True)
        use_llm_judge: Whether to use LLM judge (default: False)
        meteor_weight: Weight for METEOR score when using LLM judge (default: 0.25)
        llm_judge_weight: Weight for LLM judge score (default: 0.75)

    Returns:
        List of combined reward scores
    """
    # Compute METEOR scores
    meteor_scores = meteor_reward_batch(gt_captions, pred_captions)

    # If not using LLM judge, return METEOR only
    if not use_llm_judge:
        return meteor_scores

    # Use LLM judge
    if llm_judge is None:
        print("Warning: use_llm_judge=True but no LLM judge provided, falling back to METEOR only")
        return meteor_scores

    # Compute LLM judge scores
    llm_scores = llm_judge.judge_batch(gt_captions, pred_captions)

    # Combine scores
    combined_scores = []
    for meteor, llm in zip(meteor_scores, llm_scores):
        combined = meteor_weight * meteor + llm_judge_weight * llm
        combined_scores.append(combined)

    return combined_scores
