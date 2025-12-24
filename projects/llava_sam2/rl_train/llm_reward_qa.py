"""
LLM-based Caption Reward using QA Evaluation (DLC-Bench Style)

This module implements LLM-based caption evaluation following DLC-Bench's
question-answering approach. Instead of directly scoring captions, it:
1. Generates questions from the reference caption
2. Asks LLM to answer questions based on the generated caption
3. Computes scores based on answer correctness

Reference: describe-anything/evaluation/eval_model_outputs.py
"""

import os
import re
import time
from typing import List, Optional, Tuple, Dict, Any
from openai import OpenAI
import torch


# ============================================================================
# Question Generation Prompts
# ============================================================================

# Prompt to extract key attributes from reference caption
EXTRACT_ATTRIBUTES_PROMPT = """Analyze the following description and extract key attributes.
Output a JSON object with these fields (use null if not mentioned):
- object_type: what is the main object/subject
- color: color of the object
- shape: shape or form
- texture: texture or material
- size: size description
- position: location or position
- state: state or condition
- action: any action or movement
- context: surrounding context or background

Description: {caption}

Output ONLY valid JSON, no explanation:"""


# Question templates based on attributes
QUESTION_TEMPLATES = {
    "object_type": {
        "question": "What is the main object or subject in the description?",
        "choices": [
            ("{value}", 1),           # Correct answer
            ("Something else", 0),    # Wrong answer
            ("Not mentioned", 0),     # Not mentioned
        ]
    },
    "color": {
        "question": "What color is described?",
        "choices": [
            ("{value}", 1),
            ("A different color", -0.5),  # Wrong color is penalized
            ("Not mentioned", 0),
        ]
    },
    "texture": {
        "question": "What is the texture or material described?",
        "choices": [
            ("{value}", 1),
            ("A different texture", -0.3),
            ("Not mentioned", 0),
        ]
    },
    "size": {
        "question": "What size is described?",
        "choices": [
            ("{value}", 1),
            ("A different size", -0.3),
            ("Not mentioned", 0),
        ]
    },
    "position": {
        "question": "Where is the object located or positioned?",
        "choices": [
            ("{value}", 1),
            ("A different location", -0.3),
            ("Not mentioned", 0),
        ]
    },
}


# Prompt for answering questions (following DLC-Bench style)
QA_EVAL_PROMPT = """Answer the multiple-choice question based on the text description of an object in an image. You need to follow these rules:
1. Do not output any reasoning. Do not perform correction. Please output exactly one answer from the choices for each question. Do not repeat the question.
2. There is no need for exact matching. Please choose the closest option based on the description.

The description is:
{pred_caption}

From the description above, please answer the following question with one of the choices:
{question_text}

Output ONLY the letter (A, B, C, etc.):"""


class LLMCaptionRewardQA:
    """
    LLM-based caption reward using QA evaluation (DLC-Bench style).

    Workflow:
    1. Extract key attributes from reference caption
    2. Generate questions based on attributes
    3. Ask LLM to answer questions based on generated caption
    4. Compute score based on answers
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9100/v1",
        model: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 50,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: float = 30.0,
        verbose: bool = False,
        min_questions: int = 2,  # Minimum questions to generate per sample
        max_questions: int = 5,  # Maximum questions per sample
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "api_key")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.verbose = verbose
        self.min_questions = min_questions
        self.max_questions = max_questions

        self.client = None
        self._init_client()

        self.api_call_count = 0
        self.api_error_count = 0
        self._connection_verified = False

    def _init_client(self):
        """Initialize OpenAI client."""
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
            if self.verbose:
                print(f"[LLMCaptionRewardQA] Client initialized: {self.base_url}")
        except Exception as e:
            print(f"[LLMCaptionRewardQA] Warning: Failed to initialize client: {e}")
            self.client = None

    def verify_connection(self):
        """Verify LLM server connection."""
        if self._connection_verified:
            return True

        test_prompt = "Say 'OK' if you can read this."
        response = self._query(test_prompt)

        if response is None:
            raise ConnectionError(
                f"Cannot connect to LLM server at {self.base_url}. "
                f"Please start the vLLM server first:\n"
                f"  ./projects/llava_sam2/rl_train/start_vllm_server.sh 7 9100"
            )

        self._connection_verified = True
        if self.verbose:
            print(f"[LLMCaptionRewardQA] Server connection verified")
        return True

    def _query(self, prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        """Query the LLM with retry logic."""
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
                    max_tokens=max_tokens or self.max_tokens,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                self.api_error_count += 1
                if self.verbose:
                    print(f"[LLMCaptionRewardQA] API error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        return None

    def _extract_attributes(self, caption: str) -> Dict[str, Any]:
        """Extract key attributes from caption using LLM."""
        prompt = EXTRACT_ATTRIBUTES_PROMPT.format(caption=caption)
        response = self._query(prompt, max_tokens=200)

        if response is None:
            return {}

        # Parse JSON response
        try:
            # Clean up response
            response = response.strip()
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            response = response.strip()

            import json
            attributes = json.loads(response)
            return {k: v for k, v in attributes.items() if v is not None and v != "null"}
        except (json.JSONDecodeError, Exception) as e:
            if self.verbose:
                print(f"[LLMCaptionRewardQA] Failed to parse attributes: {e}")
            return {}

    def _generate_questions(self, attributes: Dict[str, Any]) -> List[Dict]:
        """Generate questions based on extracted attributes."""
        questions = []

        for attr_name, attr_value in attributes.items():
            if attr_name not in QUESTION_TEMPLATES:
                continue
            if attr_value is None or attr_value == "":
                continue

            template = QUESTION_TEMPLATES[attr_name]

            # Build choices with the correct answer
            choices = []
            for choice_template, score in template["choices"]:
                if "{value}" in choice_template:
                    choice_text = str(attr_value)
                else:
                    choice_text = choice_template
                choices.append((choice_text, score))

            questions.append({
                "attribute": attr_name,
                "question": template["question"],
                "choices": choices,
                "correct_value": attr_value,
            })

        # Limit number of questions
        if len(questions) > self.max_questions:
            questions = questions[:self.max_questions]

        return questions

    def _answer_question(self, pred_caption: str, question: Dict) -> Tuple[int, str]:
        """Ask LLM to answer a question based on predicted caption."""
        # Build question text with choices
        question_text = f"{question['question']}\n"
        for i, (choice, _) in enumerate(question['choices']):
            letter = chr(ord('A') + i)
            question_text += f"{letter}. {choice}\n"

        prompt = QA_EVAL_PROMPT.format(
            pred_caption=pred_caption,
            question_text=question_text.strip()
        )

        response = self._query(prompt, max_tokens=10)

        if response is None:
            return -1, "API_ERROR"

        # Parse answer
        response = response.strip().upper()
        for i in range(len(question['choices'])):
            letter = chr(ord('A') + i)
            if response.startswith(letter):
                return i, response

        # Try to find any letter in response
        for i in range(len(question['choices'])):
            letter = chr(ord('A') + i)
            if letter in response[:5]:
                return i, response

        return -1, response

    def evaluate_single(self, reference: str, hypothesis: str) -> Tuple[float, Dict]:
        """
        Evaluate a single caption pair using QA approach.

        Args:
            reference: Ground truth caption
            hypothesis: Generated caption

        Returns:
            (score, details) where score is in [0, 1] and details contains evaluation info
        """
        # Step 1: Extract attributes from reference caption
        attributes = self._extract_attributes(reference)

        if not attributes:
            # Fallback: use simple comparison if no attributes extracted
            if self.verbose:
                print("[LLMCaptionRewardQA] No attributes extracted, using fallback")
            return self._fallback_evaluate(reference, hypothesis)

        # Step 2: Generate questions
        questions = self._generate_questions(attributes)

        if len(questions) < self.min_questions:
            # Not enough questions, use fallback
            if self.verbose:
                print(f"[LLMCaptionRewardQA] Only {len(questions)} questions, using fallback")
            return self._fallback_evaluate(reference, hypothesis)

        # Step 3: Answer questions based on predicted caption
        scores = []
        details = {
            "attributes": attributes,
            "questions": [],
        }

        for q in questions:
            answer_idx, answer_text = self._answer_question(hypothesis, q)

            if answer_idx >= 0 and answer_idx < len(q['choices']):
                score = q['choices'][answer_idx][1]
            else:
                score = 0  # Invalid answer

            scores.append(score)
            details["questions"].append({
                "question": q['question'],
                "answer": answer_text,
                "score": score,
            })

        # Step 4: Compute final score (average, normalized to [0, 1])
        if scores:
            # Scores can be negative (for wrong answers), so normalize
            raw_score = sum(scores) / len(scores)
            # Map from [-0.5, 1] to [0, 1]
            final_score = (raw_score + 0.5) / 1.5
            final_score = max(0.0, min(1.0, final_score))
        else:
            final_score = 0.0

        details["raw_score"] = raw_score if scores else 0.0
        details["final_score"] = final_score
        details["num_questions"] = len(questions)

        return final_score, details

    def _fallback_evaluate(self, reference: str, hypothesis: str) -> Tuple[float, Dict]:
        """Fallback evaluation using direct comparison prompt."""
        prompt = f"""Compare these two descriptions and rate similarity from 0-10.

Reference: {reference}

Generated: {hypothesis}

Consider: Are they describing the same thing? Do key details match?
Output ONLY a number 0-10:"""

        response = self._query(prompt, max_tokens=10)

        if response is None:
            raise RuntimeError("LLM API call failed in fallback evaluation")

        try:
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0]) / 10.0
                score = max(0.0, min(1.0, score))
                return score, {"method": "fallback", "raw_response": response}
        except:
            pass

        return 0.0, {"method": "fallback", "error": "parse_failed"}

    def evaluate_batch(self, references: List[str], hypotheses: List[str]) -> List[float]:
        """Evaluate a batch of caption pairs."""
        assert len(references) == len(hypotheses)
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
        """Compute rewards for RL training."""
        scores = self.evaluate_batch(gt_captions, pred_captions)
        rewards = torch.tensor(scores, dtype=torch.float32, device=device)
        return rewards

    def get_stats(self) -> dict:
        """Get API call statistics."""
        return {
            "api_call_count": self.api_call_count,
            "api_error_count": self.api_error_count,
        }


def test_qa_reward():
    """Test the QA-style LLM reward module."""
    print("=" * 70)
    print("Testing LLM Caption Reward (QA Style)")
    print("=" * 70)

    reward_fn = LLMCaptionRewardQA(
        base_url="http://localhost:9100/v1",
        verbose=True
    )

    # Test caption pair
    reference = "A red apple sitting on a wooden table with natural lighting. The apple has a smooth, shiny surface."
    hypothesis = "A red fruit on a wooden surface. It appears to be an apple with a glossy texture."

    print(f"\nReference: {reference}")
    print(f"Hypothesis: {hypothesis}")
    print()

    try:
        reward_fn.verify_connection()
        score, details = reward_fn.evaluate_single(reference, hypothesis)
        print(f"\nScore: {score:.3f}")
        print(f"Details: {details}")
    except ConnectionError as e:
        print(f"Connection Error: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    test_qa_reward()
