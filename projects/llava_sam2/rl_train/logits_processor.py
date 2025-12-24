"""
Logits processors for numerical stability during generation.

These processors help prevent NaN/inf issues during sampling by:
1. Clipping extreme logit values
2. Detecting and fixing NaN/inf values
3. Ensuring numerical stability in probability computation
"""

import torch
from transformers import LogitsProcessor
import numpy as np


class NumericalStabilityLogitsProcessor(LogitsProcessor):
    """
    Logits processor that ensures numerical stability during generation.

    This processor:
    1. Clips logits to a reasonable range to prevent overflow/underflow
    2. Detects and replaces NaN/inf values
    3. Ensures probabilities can be computed safely

    Args:
        clip_value: Maximum absolute value for logits (default: 30.0)
            - This prevents exp(logits) from overflowing (exp(30) ≈ 1e13)
        min_prob: Minimum probability for any token (default: 1e-8)
        verbose: Whether to print warnings when NaN/inf are detected
    """

    def __init__(self, clip_value: float = 30.0, min_prob: float = 1e-8, verbose: bool = True):
        self.clip_value = clip_value
        self.min_prob = min_prob
        self.verbose = verbose
        self.nan_count = 0
        self.inf_count = 0
        self.clip_count = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits to ensure numerical stability.

        Args:
            input_ids: (batch_size, seq_len) - Generated tokens so far
            scores: (batch_size, vocab_size) - Raw logits from the model

        Returns:
            Processed logits that are numerically stable
        """
        # Step 1: Detect NaN/inf values
        has_nan = torch.isnan(scores).any()
        has_inf = torch.isinf(scores).any()

        if has_nan:
            self.nan_count += 1
            if self.verbose:
                print(f"⚠ WARNING: NaN detected in logits (occurrence #{self.nan_count})")
                print(f"  Input shape: {input_ids.shape}, Logits shape: {scores.shape}")
            # Replace NaN with very negative value (will become near-zero probability)
            scores = torch.nan_to_num(scores, nan=-100.0, posinf=self.clip_value, neginf=-self.clip_value)

        if has_inf:
            self.inf_count += 1
            if self.verbose:
                print(f"⚠ WARNING: Inf detected in logits (occurrence #{self.inf_count})")
                print(f"  Input shape: {input_ids.shape}, Logits shape: {scores.shape}")
            # Replace inf with clip value
            scores = torch.nan_to_num(scores, nan=-100.0, posinf=self.clip_value, neginf=-self.clip_value)

        # Step 2: Clip logits to prevent overflow in softmax/exp
        max_logit = scores.abs().max().item()
        if max_logit > self.clip_value:
            self.clip_count += 1
            if self.verbose and self.clip_count <= 5:  # Only print first 5 times
                print(f"⚠ Clipping logits: max={max_logit:.2f} -> {self.clip_value}")
            scores = torch.clamp(scores, -self.clip_value, self.clip_value)

        # Step 3: Ensure numerical stability in probability computation
        # Subtract max for numerical stability (standard softmax trick)
        scores_max = scores.max(dim=-1, keepdim=True)[0]
        scores = scores - scores_max

        # Step 4: Apply minimum probability floor
        # Convert to log probabilities, apply floor, convert back
        log_probs = torch.log_softmax(scores, dim=-1)
        log_probs = torch.clamp(log_probs, min=np.log(self.min_prob))

        # Convert back to logits (unnormalized)
        # We return log_probs directly as they're already stabilized
        # (transformers will apply softmax again, but that's ok)
        scores = log_probs

        return scores

    def reset_stats(self):
        """Reset statistics counters."""
        self.nan_count = 0
        self.inf_count = 0
        self.clip_count = 0

    def get_stats(self):
        """Get statistics about numerical issues encountered."""
        return {
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'clip_count': self.clip_count,
        }


class TemperatureLogitsWarper(LogitsProcessor):
    """
    Temperature warping with numerical stability.

    Unlike the default temperature warping, this version:
    1. Ensures temperature is never too small (min 0.1)
    2. Clips logits after division to prevent overflow
    3. Detects numerical issues

    Args:
        temperature: Sampling temperature (default: 1.0)
        min_temperature: Minimum allowed temperature (default: 0.1)
    """

    def __init__(self, temperature: float = 1.0, min_temperature: float = 0.1):
        self.temperature = max(temperature, min_temperature)
        self.min_temperature = min_temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Apply temperature scaling."""
        # Apply temperature
        scores = scores / self.temperature

        # Clip to prevent overflow
        scores = torch.clamp(scores, -100.0, 100.0)

        return scores
