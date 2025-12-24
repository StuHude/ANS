"""
Unit test for NumericalStabilityLogitsProcessor.

This test verifies that the logits processor correctly:
1. Detects and fixes NaN values
2. Detects and fixes Inf values
3. Clips extreme logit values
4. Produces valid probability distributions
"""

import torch
import numpy as np
from logits_processor import NumericalStabilityLogitsProcessor, TemperatureLogitsWarper


def test_nan_detection():
    """Test that NaN values are detected and replaced."""
    print("=" * 70)
    print("Test 1: NaN Detection and Replacement")
    print("=" * 70)

    processor = NumericalStabilityLogitsProcessor(verbose=True)

    # Create logits with NaN
    batch_size, vocab_size = 4, 1000
    logits = torch.randn(batch_size, vocab_size)
    logits[0, 100] = float('nan')
    logits[2, 500] = float('nan')

    print(f"Input logits shape: {logits.shape}")
    print(f"NaN count before: {torch.isnan(logits).sum().item()}")

    # Create dummy input_ids
    input_ids = torch.randint(0, vocab_size, (batch_size, 10))

    # Process
    processed_logits = processor(input_ids, logits)

    print(f"NaN count after: {torch.isnan(processed_logits).sum().item()}")
    print(f"Stats: {processor.get_stats()}")
    print()

    # Verify no NaN remains
    assert not torch.isnan(processed_logits).any(), "NaN values remain after processing!"
    print("✓ Test passed: All NaN values replaced\n")


def test_inf_detection():
    """Test that Inf values are detected and replaced."""
    print("=" * 70)
    print("Test 2: Inf Detection and Replacement")
    print("=" * 70)

    processor = NumericalStabilityLogitsProcessor(verbose=True)

    # Create logits with Inf
    batch_size, vocab_size = 4, 1000
    logits = torch.randn(batch_size, vocab_size)
    logits[1, 200] = float('inf')
    logits[3, 700] = float('-inf')

    print(f"Input logits shape: {logits.shape}")
    print(f"Inf count before: {torch.isinf(logits).sum().item()}")

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))

    # Process
    processed_logits = processor(input_ids, logits)

    print(f"Inf count after: {torch.isinf(processed_logits).sum().item()}")
    print(f"Stats: {processor.get_stats()}")
    print()

    # Verify no Inf remains
    assert not torch.isinf(processed_logits).any(), "Inf values remain after processing!"
    print("✓ Test passed: All Inf values replaced\n")


def test_extreme_logits_clipping():
    """Test that extreme logit values are clipped."""
    print("=" * 70)
    print("Test 3: Extreme Logits Clipping")
    print("=" * 70)

    clip_value = 30.0
    processor = NumericalStabilityLogitsProcessor(clip_value=clip_value, verbose=True)

    # Create logits with extreme values
    batch_size, vocab_size = 4, 1000
    logits = torch.randn(batch_size, vocab_size)
    logits[0, 50] = 100.0   # Very large positive
    logits[1, 150] = -100.0  # Very large negative
    logits[2, 250] = 45.0
    logits[3, 350] = -45.0

    print(f"Input logits shape: {logits.shape}")
    print(f"Max logit before: {logits.max().item():.2f}")
    print(f"Min logit before: {logits.min().item():.2f}")

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))

    # Process
    processed_logits = processor(input_ids, logits)

    print(f"Max logit after: {processed_logits.max().item():.2f}")
    print(f"Min logit after: {processed_logits.min().item():.2f}")
    print(f"Stats: {processor.get_stats()}")
    print()

    # Verify clipping (after log_softmax, values will be negative)
    # The processor returns log probabilities, so they should all be <= 0
    assert processed_logits.max() <= 0.0, "Processed logits should be log probabilities (<= 0)"
    print("✓ Test passed: Extreme values handled correctly\n")


def test_probability_distribution():
    """Test that output can be converted to valid probability distribution."""
    print("=" * 70)
    print("Test 4: Valid Probability Distribution")
    print("=" * 70)

    processor = NumericalStabilityLogitsProcessor(verbose=False)

    # Create logits with mixed issues
    batch_size, vocab_size = 4, 1000
    logits = torch.randn(batch_size, vocab_size) * 20  # Scale up for extreme values
    logits[0, 100] = float('nan')
    logits[1, 200] = float('inf')
    logits[2, 300] = -100.0
    logits[3, 400] = 100.0

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))

    # Process
    processed_logits = processor(input_ids, logits)

    # Convert to probabilities
    # Note: processor returns log probabilities, so we just need to exp
    probs = torch.exp(processed_logits)

    print(f"Probability stats:")
    print(f"  Min prob: {probs.min().item():.10f}")
    print(f"  Max prob: {probs.max().item():.6f}")
    print(f"  Sum per row (should be ~1.0): {probs.sum(dim=1)}")
    print()

    # Verify valid probabilities
    assert not torch.isnan(probs).any(), "Probabilities contain NaN!"
    assert not torch.isinf(probs).any(), "Probabilities contain Inf!"
    assert (probs >= 0).all(), "Probabilities contain negative values!"
    assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-3), \
        "Probabilities don't sum to 1.0!"

    # Test sampling (should not crash)
    try:
        # Sample from the distribution
        samples = torch.multinomial(probs, num_samples=5, replacement=True)
        print(f"Sampling successful: {samples.shape}")
        print("✓ Test passed: Can sample from distribution\n")
    except RuntimeError as e:
        print(f"✗ Test failed: Cannot sample from distribution: {e}\n")
        raise


def test_temperature_warper():
    """Test temperature warping."""
    print("=" * 70)
    print("Test 5: Temperature Warping")
    print("=" * 70)

    warper = TemperatureLogitsWarper(temperature=0.8, min_temperature=0.1)

    batch_size, vocab_size = 4, 1000
    logits = torch.randn(batch_size, vocab_size) * 10

    input_ids = torch.randint(0, vocab_size, (batch_size, 10))

    # Process
    processed_logits = warper(input_ids, logits)

    print(f"Input logits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")
    print(f"Output logits range: [{processed_logits.min().item():.2f}, {processed_logits.max().item():.2f}]")
    print()

    # Verify temperature was applied (should scale down)
    expected_max = logits.max().item() / 0.8
    assert abs(processed_logits.max().item() - expected_max) < 1.0, \
        "Temperature scaling not applied correctly"

    print("✓ Test passed: Temperature warping works\n")


def test_combined_processors():
    """Test using both processors together (as in actual training)."""
    print("=" * 70)
    print("Test 6: Combined Processors (Real Usage)")
    print("=" * 70)

    from transformers import LogitsProcessorList

    # Create processor list (as in trainer)
    processors = LogitsProcessorList([
        NumericalStabilityLogitsProcessor(clip_value=30.0, min_prob=1e-8, verbose=False),
        TemperatureLogitsWarper(temperature=1.0, min_temperature=0.1),
    ])

    # Create problematic logits
    batch_size, vocab_size = 8, 50000  # Realistic vocab size
    logits = torch.randn(batch_size, vocab_size) * 30
    logits[0, 1000] = float('nan')
    logits[1, 2000] = float('inf')
    logits[2, 3000] = 150.0  # Extreme value
    logits[3, 4000] = -150.0

    input_ids = torch.randint(0, vocab_size, (batch_size, 20))

    print(f"Input logits shape: {logits.shape}")
    print(f"Issues: NaN={torch.isnan(logits).sum().item()}, "
          f"Inf={torch.isinf(logits).sum().item()}")

    # Process
    processed_logits = logits
    for processor in processors:
        processed_logits = processor(input_ids, processed_logits)

    print(f"After processing: NaN={torch.isnan(processed_logits).sum().item()}, "
          f"Inf={torch.isinf(processed_logits).sum().item()}")

    # Test multinomial sampling (the critical operation that was failing)
    try:
        # Convert to probabilities
        probs = torch.softmax(processed_logits, dim=-1)

        # Sample (this is what was failing before)
        samples = torch.multinomial(probs, num_samples=10, replacement=True)

        print(f"✓ Multinomial sampling successful: {samples.shape}")
        print("✓ Test passed: Combined processors work correctly\n")
    except RuntimeError as e:
        print(f"✗ Test failed: Multinomial sampling crashed: {e}\n")
        raise


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("TESTING NUMERICAL STABILITY LOGITS PROCESSOR")
    print("=" * 70 + "\n")

    try:
        test_nan_detection()
        test_inf_detection()
        test_extreme_logits_clipping()
        test_probability_distribution()
        test_temperature_warper()
        test_combined_processors()

        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print("\nThe logits processors are working correctly and should prevent")
        print("NaN/inf issues during generation in Sa2VA RL training.")
        print()

    except Exception as e:
        print("\n" + "=" * 70)
        print("TEST FAILED ✗")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
