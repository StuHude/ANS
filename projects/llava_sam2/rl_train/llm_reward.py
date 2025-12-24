"""
LLM-based Caption Reward for RL Training

This module implements LLM-based caption evaluation following the describe-anything
project's approach. It uses vLLM to serve Llama-3.1-8B-Instruct and evaluates
caption quality through the OpenAI-compatible API.

Usage:
1. Start vLLM server:
   CUDA_VISIBLE_DEVICES=7 vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct \
       --tensor-parallel-size 1 --port 9100 --max-model-len 4096

2. Use in training:
   from llm_reward import LLMCaptionReward
   reward_fn = LLMCaptionReward(base_url="http://localhost:9100/v1")
   rewards = reward_fn.compute_rewards(gt_captions, pred_captions)
"""

import os
import re
import time
from typing import List, Optional, Tuple
from openai import OpenAI
import torch


# ============================================================================
# Prompt Templates (following describe-anything evaluation style)
# ============================================================================

# Direct scoring prompt - asks LLM to rate caption similarity/quality
CAPTION_EVAL_PROMPT = """You are an expert evaluator of image region descriptions. Your task is to rate how well a generated description matches a reference description.

Reference description (ground truth):
{reference}

Generated description (to evaluate):
{hypothesis}

Rate the generated description on a scale from 0 to 10 based on:
1. **Semantic Accuracy**: Does it describe the same object/region correctly?
2. **Detail Coverage**: Does it capture the key details from the reference?
3. **Factual Correctness**: Are the stated facts consistent with the reference?

IMPORTANT RULES:
- Exact word matching is NOT required
- Focus on meaning and content, not wording
- A score of 10 means perfect semantic match
- A score of 0 means completely wrong or irrelevant
- Partial matches should get proportional scores (e.g., 5-7)

Output ONLY a single number between 0 and 10 (decimals allowed, e.g., 7.5).
Do NOT include any explanation or other text.

Score:"""


# Alternative: Multiple-choice style evaluation (more similar to DLC-Bench)
CAPTION_EVAL_MCQ_PROMPT = """Based on the reference description, evaluate the generated description.

Reference description:
{reference}

Generated description:
{hypothesis}

How well does the generated description match the reference?
A. Excellent match - captures all key information correctly (Score: 10)
B. Good match - captures most key information with minor omissions (Score: 7-8)
C. Partial match - captures some information but missing important details (Score: 4-6)
D. Poor match - significantly different or incorrect information (Score: 1-3)
E. No match - completely wrong or irrelevant (Score: 0)

Output ONLY the letter (A, B, C, D, or E):"""


class LLMCaptionReward:
    """
    LLM-based caption reward calculator for RL training.

    Uses vLLM server with OpenAI-compatible API to evaluate caption quality.
    """

    # Score mapping for MCQ-style evaluation
    MCQ_SCORES = {
        'A': 1.0,
        'B': 0.75,
        'C': 0.5,
        'D': 0.2,
        'E': 0.0,
    }

    def __init__(
        self,
        base_url: str = "http://localhost:9100/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 10,
        eval_mode: str = "score",  # "score" or "mcq"
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        """
        Initialize LLM caption reward calculator.

        Args:
            base_url: vLLM server URL (OpenAI-compatible API)
            model: Model name/ID
            api_key: API key (use "api_key" for local vLLM)
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Max tokens for response
            eval_mode: "score" for direct 0-10 scoring, "mcq" for multiple choice
            max_retries: Number of retries on API failure
            retry_delay: Delay between retries in seconds
            timeout: Request timeout in seconds
            verbose: Print debug information
        """
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "api_key")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.eval_mode = eval_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verbose = verbose

        # Initialize OpenAI client
        self.client = None
        self._init_client()

        # API call counter for monitoring
        self.api_call_count = 0
        self.api_error_count = 0

    def _init_client(self):
        """Initialize OpenAI client with retry logic."""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            if self.verbose:
                print(f"[LLMCaptionReward] Client initialized: {self.base_url}")
        except Exception as e:
            print(f"[LLMCaptionReward] Warning: Failed to initialize client: {e}")
            self.client = None

    def _query(self, prompt: str) -> Optional[str]:
        """
        Query the LLM with retry logic.

        Args:
            prompt: The prompt to send to the LLM

        Returns:
            Response string or None if failed
        """
        if self.client is None:
            self._init_client()
            if self.client is None:
                return None

        for attempt in range(self.max_retries):
            try:
                self.api_call_count += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                self.api_error_count += 1
                if self.verbose:
                    print(f"[LLMCaptionReward] API error (attempt {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def _parse_score(self, response: str) -> float:
        """
        Parse score from LLM response.

        Args:
            response: Raw LLM response

        Returns:
            Normalized score between 0 and 1
        """
        if response is None:
            return 0.0

        response = response.strip().lower()

        if self.eval_mode == "mcq":
            # Parse multiple choice answer
            for letter in "abcde":
                if letter in response[:5]:  # Check first few characters
                    return self.MCQ_SCORES.get(letter.upper(), 0.0)
            return 0.0
        else:
            # Parse numeric score
            try:
                # Extract first number from response
                numbers = re.findall(r'\d+\.?\d*', response)
                if numbers:
                    score = float(numbers[0])
                    # Normalize to [0, 1]
                    score = max(0.0, min(10.0, score)) / 10.0
                    return score
            except (ValueError, IndexError):
                pass

            return 0.0

    def evaluate_single(self, reference: str, hypothesis: str) -> Tuple[float, Optional[str]]:
        """
        Evaluate a single caption pair.

        Args:
            reference: Ground truth caption
            hypothesis: Generated caption

        Returns:
            Tuple of (score, raw_response)
        """
        if self.eval_mode == "mcq":
            prompt = CAPTION_EVAL_MCQ_PROMPT.format(
                reference=reference,
                hypothesis=hypothesis
            )
        else:
            prompt = CAPTION_EVAL_PROMPT.format(
                reference=reference,
                hypothesis=hypothesis
            )

        response = self._query(prompt)
        score = self._parse_score(response)

        return score, response

    def evaluate_batch(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """
        Evaluate a batch of caption pairs.

        Args:
            references: List of ground truth captions
            hypotheses: List of generated captions

        Returns:
            List of scores between 0 and 1
        """
        assert len(references) == len(hypotheses), \
            f"Batch size mismatch: {len(references)} vs {len(hypotheses)}"

        scores = []
        for ref, hyp in zip(references, hypotheses):
            score, _ = self.evaluate_single(ref, hyp)
            scores.append(score)

        return scores

    def compute_rewards(
        self,
        gt_captions: List[str],
        pred_captions: List[str],
        device: str = "cuda"
    ) -> torch.Tensor:
        """
        Compute rewards for RL training.

        Args:
            gt_captions: List of ground truth captions
            pred_captions: List of predicted captions
            device: Device for output tensor

        Returns:
            Tensor of rewards with shape (batch_size,)
        """
        scores = self.evaluate_batch(gt_captions, pred_captions)
        rewards = torch.tensor(scores, dtype=torch.float32, device=device)
        return rewards

    def get_stats(self) -> dict:
        """Get API call statistics."""
        return {
            "api_call_count": self.api_call_count,
            "api_error_count": self.api_error_count,
            "error_rate": self.api_error_count / max(1, self.api_call_count),
        }

    def reset_stats(self):
        """Reset API call statistics."""
        self.api_call_count = 0
        self.api_error_count = 0


class LLMCaptionRewardStrict(LLMCaptionReward):
    """
    Strict LLM caption reward - raises error if LLM API fails.

    No fallback to METEOR. If LLM API is unavailable, raises an exception.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._connection_verified = False

    def verify_connection(self):
        """Verify LLM server connection. Raises error if unavailable."""
        if self._connection_verified:
            return True

        test_prompt = "Say 'OK' if you can read this."
        response = self._query(test_prompt)

        if response is None:
            raise ConnectionError(
                f"Cannot connect to LLM server at {self.base_url}. "
                f"Please start the vLLM server first:\n"
                f"  ./projects/llava_sam2/rl_train/start_vllm_server.sh 7 9100\n"
                f"Then verify it's running:\n"
                f"  curl {self.base_url}/models"
            )

        self._connection_verified = True
        if self.verbose:
            print(f"[LLMCaptionReward] Server connection verified: {self.base_url}")
        return True

    def evaluate_single(self, reference: str, hypothesis: str) -> Tuple[float, Optional[str]]:
        """Evaluate with LLM. Raises error if API fails."""
        score, response = super().evaluate_single(reference, hypothesis)

        if response is None:
            raise RuntimeError(
                f"LLM API call failed after {self.max_retries} retries. "
                f"Server: {self.base_url}, Model: {self.model}. "
                f"Please check the vLLM server status."
            )

        return score, response


def test_llm_reward():
    """Test the LLM reward module."""
    print("=" * 70)
    print("Testing LLM Caption Reward")
    print("=" * 70)

    # Test initialization
    reward_fn = LLMCaptionReward(
        base_url="http://localhost:9100/v1",
        verbose=True
    )

    # Test captions
    test_cases = [
        {
            "reference": "A red apple sitting on a wooden table with natural lighting.",
            "hypothesis": "A red apple on a wooden surface in good lighting conditions.",
            "expected": "high"  # Should be high similarity
        },
        {
            "reference": "A red apple sitting on a wooden table.",
            "hypothesis": "A blue car parked on the street.",
            "expected": "low"  # Should be low similarity
        },
        {
            "reference": "The image shows a detailed view of tree bark with horizontal ridges.",
            "hypothesis": "Tree bark with visible ridges and natural patterns.",
            "expected": "high"
        },
    ]

    print("\nRunning test cases...")
    for i, tc in enumerate(test_cases):
        score, response = reward_fn.evaluate_single(tc["reference"], tc["hypothesis"])
        print(f"\nTest {i+1}:")
        print(f"  Reference: {tc['reference'][:50]}...")
        print(f"  Hypothesis: {tc['hypothesis'][:50]}...")
        print(f"  Score: {score:.3f} (expected: {tc['expected']})")
        print(f"  Response: {response}")

    # Print stats
    stats = reward_fn.get_stats()
    print(f"\nAPI Stats: {stats}")

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    test_llm_reward()
