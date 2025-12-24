# NaN/Inf Fix for Sa2VA RL Training

## Problem Summary

During Sa2VA RL training with GRPO, the model encountered `RuntimeError: probability tensor contains either 'inf', 'nan' or element < 0` during the generation phase. This error occurred in `torch.multinomial()` when sampling from the probability distribution.

## Root Cause Analysis

The NaN/inf issue was caused by numerical instability in the logits during generation:

1. **Extreme logit values**: Without clipping, logits could become very large (>100) or very small (<-100)
2. **Softmax overflow**: When computing `exp(logits)` in softmax, extreme values cause overflow (inf) or underflow (0)
3. **Division issues**: In probability normalization, division by very small numbers can produce NaN
4. **Gradient explosion**: Large gradients in early training steps can destabilize model weights

## Implemented Fixes

### 1. Numerical Stability Logits Processor (`logits_processor.py`)

Created `NumericalStabilityLogitsProcessor` that:
- **Detects NaN/inf**: Monitors logits for numerical issues
- **Clips extreme values**: Limits logits to [-30, 30] range to prevent overflow
- **Applies softmax stabilization**: Uses log-space computation with max subtraction trick
- **Enforces minimum probability**: Ensures no token has probability below 1e-8

**Location**: `projects/llava_sam2/rl_train/logits_processor.py`

### 2. Temperature Warping with Safety (`logits_processor.py`)

Created `TemperatureLogitsWarper` that:
- **Prevents too-small temperature**: Enforces minimum temperature of 0.1
- **Clips after division**: Prevents overflow from temperature scaling
- **Monitors for issues**: Detects numerical problems early

### 3. Gradient Monitoring Callback (`training_callbacks.py`)

Created `GradientMonitorCallback` that:
- **Checks gradients every step**: Detects NaN/inf in gradients immediately
- **Tracks statistics**: Logs gradient norms and identifies problematic parameters
- **Provides diagnostics**: Shows which layers have gradient issues
- **Optional halt**: Can stop training if NaN is detected

**Location**: `projects/llava_sam2/rl_train/training_callbacks.py`

### 4. Enhanced Trainer Configuration (`train_sa2va_rl.py`)

Updated GRPO config with:
- `max_grad_norm=1.0`: Gradient clipping to prevent explosion
- `warmup_steps=10`: Gradual learning rate warmup for stability
- `bf16=True`: Use bfloat16 instead of fp16 for better numerical range
- `fp16=False`: Disable fp16 which is less stable than bf16

### 5. Integrated Logits Processor in Trainer (`sa2va_grpo_trainer.py`)

Modified trainer to:
- Initialize logits processors in `__init__`
- Pass processors to `model.generate()` call
- Log when numerical issues are detected and corrected

## Key Code Changes

### 1. Logits Processor Integration

```python
# In sa2va_grpo_trainer.py __init__
self.logits_processor = LogitsProcessorList([
    NumericalStabilityLogitsProcessor(clip_value=30.0, min_prob=1e-8, verbose=True),
    TemperatureLogitsWarper(temperature=1.0, min_temperature=0.1),
])
```

### 2. Generation with Stability

```python
# In sa2va_grpo_trainer.py compute_loss
prompt_completion_ids = unwrapped_model.generate(
    input_ids=prompt_ids,
    pixel_values=pixel_values,
    prompt_masks=prompt_masks,
    vp_overall_mask=vp_overall_mask,
    generation_config=self.generation_config,
    attention_mask=prompt_mask,
    logits_processor=self.logits_processor,  # ← KEY FIX
)
```

### 3. Gradient Monitoring

```python
# In train_sa2va_rl.py
callbacks = [
    GradientMonitorCallback(
        check_every_n_steps=1,
        log_every_n_steps=10,
        halt_on_nan=False,
    ),
]
```

## Testing

Run the test script to verify fixes:

```bash
bash projects/llava_sam2/rl_train/test_nan_fix.sh
```

This will:
1. Run a short training session (few steps)
2. Monitor for NaN/inf in both logits and gradients
3. Report statistics on numerical stability
4. Verify gradient clipping is working

Expected output:
- **Best case**: No NaN/inf detected at all
- **Acceptable**: NaN/inf detected but handled by processors (especially in first few steps)
- **Failure**: NaN/inf causes training to crash

## Monitoring During Training

During training, you'll see:

### Logits Processor Warnings (if issues detected):
```
⚠ WARNING: NaN detected in logits (occurrence #1)
  Input shape: torch.Size([8, 512]), Logits shape: torch.Size([8, 151936])
⚠ Clipping logits: max=45.23 -> 30.0
```

### Gradient Monitor Reports (every 10 steps):
```
[Step 10] Gradient Statistics:
  Mean grad norm: 0.823456
  Max grad norm: 1.234567
  Min grad norm: 0.123456
  NaN count: 0, Inf count: 0
```

### Gradient Issues (if detected):
```
======================================================================
⚠ GRADIENT ISSUE DETECTED at step 42
======================================================================
  NaN detected: True (total count: 1)
  Inf detected: False (total count: 0)
  Total gradient norm: inf

Parameters with issues:
  - language_model.base_model.model.model.layers.23.mlp.gate_proj.lora_B.default.weight (NaN)
======================================================================
```

## What to Do If Issues Persist

If NaN/inf errors still occur after these fixes:

### 1. Reduce Learning Rate
```bash
python train_sa2va_rl.py --learning_rate 5e-6  # Half of current 1e-5
```

### 2. Increase Gradient Clipping
Edit `train_sa2va_rl.py`:
```python
max_grad_norm=0.5,  # More aggressive clipping
```

### 3. Tighten Logits Clipping
Edit `sa2va_grpo_trainer.py`:
```python
NumericalStabilityLogitsProcessor(clip_value=20.0, ...)  # Tighter clipping
```

### 4. Add Activation Monitoring
Uncomment in `train_sa2va_rl.py`:
```python
callbacks = [
    GradientMonitorCallback(...),
    ActivationMonitorCallback(check_every_n_steps=50),  # ← Enable this
]
```

### 5. Enable Halt on NaN
For debugging, stop immediately when NaN detected:
```python
GradientMonitorCallback(
    check_every_n_steps=1,
    log_every_n_steps=10,
    halt_on_nan=True,  # ← Change to True
)
```

## Technical Details

### Why Logits Clipping Works

The softmax function is:
```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Problems:
- If `x_i > 88`: `exp(x_i)` overflows to inf (float32)
- If `x_i < -88`: `exp(x_i)` underflows to 0

Solution:
- Clip logits to [-30, 30] range
- Use stable softmax: `softmax(x - max(x))`
- Work in log space when possible

### Why BFloat16 is Better Than FP16

- **FP16 range**: ±65,504 (overflows easily)
- **BF16 range**: ±3.4×10³⁸ (same as FP32)
- **BF16 precision**: Lower than FP16, but rarely matters for NN
- **Gradient stability**: BF16 handles extreme values better

### Gradient Clipping Mathematics

Given gradients `g_1, g_2, ..., g_n`, gradient clipping ensures:
```
||g||_2 <= max_grad_norm
```

If `||g||_2 > max_grad_norm`:
```
g_i' = g_i * (max_grad_norm / ||g||_2)
```

This prevents any single large gradient from destabilizing training.

## Files Modified

1. `projects/llava_sam2/rl_train/logits_processor.py` (NEW)
2. `projects/llava_sam2/rl_train/gradient_monitor.py` (NEW)
3. `projects/llava_sam2/rl_train/training_callbacks.py` (NEW)
4. `projects/llava_sam2/rl_train/sa2va_grpo_trainer.py` (MODIFIED)
5. `projects/llava_sam2/rl_train/train_sa2va_rl.py` (MODIFIED)
6. `projects/llava_sam2/rl_train/test_nan_fix.sh` (NEW)

## References

- Softmax stability: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
- Gradient clipping: Pascanu et al., "On the difficulty of training RNNs" (2013)
- BFloat16: https://en.wikipedia.org/wiki/Bfloat16_floating-point_format
- Numerical stability in deep learning: https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/
