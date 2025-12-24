# Mask Stacking Fix for Batch Size > 1

## Problem

When `batch_size > 1`, the training code would crash with an error when trying to stack `pred_masks_list` because different images may have different original sizes, resulting in masks with incompatible dimensions.

### Root Cause

In `sa2va_dual_loop_trainer_v2.py`, line 310 used:
```python
pred_masks = torch.stack(pred_masks_list, dim=0)  # (B, H, W)
```

This fails when masks in `pred_masks_list` have different shapes, e.g.:
- Image 1: mask shape (400, 600)
- Image 2: mask shape (512, 512)
- Image 3: mask shape (300, 450)

`torch.stack()` requires all tensors to have the same shape.

## Solution

Added a new helper method `_stack_masks_with_resize()` that:

1. **Resizes all masks to a unified size** (default: 448×448)
2. **Handles various input formats**: 2D, 3D, 4D tensors
3. **Preserves data types**: Correctly handles bool, float, and int masks
4. **Supports custom target sizes**: Can specify different output dimensions

### Implementation

**File**: `projects/llava_sam2/rl_train/sa2va_dual_loop_trainer_v2.py`

**Changes**:

1. **Line 312** - Replaced direct `torch.stack()` with `_stack_masks_with_resize()`:
   ```python
   # Old (broken for batch_size > 1):
   pred_masks = torch.stack(pred_masks_list, dim=0)

   # New (works for any batch size):
   pred_masks = self._stack_masks_with_resize(pred_masks_list, device)
   ```

2. **Lines 459-525** - Added new method:
   ```python
   def _stack_masks_with_resize(self, masks_list, device, target_size=448):
       """Stack masks with different sizes by resizing them to a unified size."""
       # Resize each mask to target_size using bilinear interpolation
       # Then stack all masks into (B, H, W) tensor
   ```

3. **Lines 535-566** - Updated `_generate_captions_from_masks_ema()` to handle both tensors and lists

## Features

- **Flexible input**: Accepts masks of any size
- **Type preservation**: Maintains bool/int/float dtypes appropriately
- **Device handling**: Ensures all masks are on the correct device
- **Configurable**: Target size can be customized via parameter
- **Robust**: Handles edge cases like 3D/4D inputs

## Testing

Created `test_mask_stacking.py` to verify:
- ✓ Different sized masks (400×600, 512×512, 300×450)
- ✓ Same sized masks (448×448, 448×448)
- ✓ Different data types (bool, float, int)
- ✓ Custom target sizes (256×256)

All tests pass successfully.

## Usage

The fix is automatically applied in the training loop. No changes needed to existing training scripts.

When using custom implementations of `_extract_mask_from_generation()`, masks can now be any size - they will be automatically resized before stacking.

## Performance

- Minimal overhead: Only resizes masks that don't match target size
- Uses efficient `F.interpolate()` with bilinear mode
- Maintains gradient flow for backpropagation
