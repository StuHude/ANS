"""
Test script to verify mask stacking with different sizes works correctly.
"""

import torch
import torch.nn.functional as F


def _stack_masks_with_resize(masks_list, device, target_size=448):
    """
    Stack masks with different sizes by resizing them to a unified size.

    Args:
        masks_list: List of masks with potentially different sizes [(H1, W1), (H2, W2), ...]
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
        # Ensure mask is a tensor
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, device=device)

        # Move to correct device
        mask = mask.to(device)

        # Handle different input shapes
        if mask.ndim == 2:
            # (H, W) -> (1, 1, H, W) for interpolate
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.ndim == 3:
            # (1, H, W) -> (1, 1, H, W)
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
            # Convert back to bool/int type
            if mask.dtype == torch.bool:
                mask_resized = mask_resized > 0.5
            else:
                mask_resized = mask_resized.round()
            mask = mask_resized

        # Remove extra dimensions: (1, 1, H, W) -> (H, W)
        mask = mask.squeeze(0).squeeze(0)
        resized_masks.append(mask)

    # Stack all masks
    stacked_masks = torch.stack(resized_masks, dim=0)  # (B, H, W)
    return stacked_masks


def test_mask_stacking():
    """Test mask stacking with different sizes."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    # Test case 1: Different sized masks (simulating batch_size > 1)
    print("\n=== Test 1: Different sized masks ===")
    mask1 = torch.rand(400, 600) > 0.5  # (400, 600)
    mask2 = torch.rand(512, 512) > 0.5  # (512, 512)
    mask3 = torch.rand(300, 450) > 0.5  # (300, 450)

    masks_list = [mask1, mask2, mask3]
    print(f"Input mask shapes: {[m.shape for m in masks_list]}")

    try:
        # This would fail with torch.stack
        # stacked = torch.stack(masks_list)

        # But should work with our function
        stacked = _stack_masks_with_resize(masks_list, device, target_size=448)
        print(f"✓ Successfully stacked! Output shape: {stacked.shape}")
        assert stacked.shape == (3, 448, 448), f"Expected (3, 448, 448), got {stacked.shape}"
        print("✓ Shape is correct!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test case 2: Same sized masks (should still work)
    print("\n=== Test 2: Same sized masks ===")
    mask1 = torch.rand(448, 448) > 0.5
    mask2 = torch.rand(448, 448) > 0.5
    masks_list = [mask1, mask2]
    print(f"Input mask shapes: {[m.shape for m in masks_list]}")

    try:
        stacked = _stack_masks_with_resize(masks_list, device, target_size=448)
        print(f"✓ Successfully stacked! Output shape: {stacked.shape}")
        assert stacked.shape == (2, 448, 448), f"Expected (2, 448, 448), got {stacked.shape}"
        print("✓ Shape is correct!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test case 3: Different dtypes
    print("\n=== Test 3: Different data types ===")
    mask1 = torch.rand(400, 600) > 0.5  # bool
    mask2 = (torch.rand(512, 512) > 0.5).float()  # float
    mask3 = (torch.rand(300, 450) > 0.5).int()  # int
    masks_list = [mask1, mask2, mask3]
    print(f"Input mask dtypes: {[m.dtype for m in masks_list]}")

    try:
        stacked = _stack_masks_with_resize(masks_list, device, target_size=448)
        print(f"✓ Successfully stacked! Output shape: {stacked.shape}, dtype: {stacked.dtype}")
        assert stacked.shape == (3, 448, 448), f"Expected (3, 448, 448), got {stacked.shape}"
        print("✓ Shape is correct!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    # Test case 4: Custom target size
    print("\n=== Test 4: Custom target size ===")
    mask1 = torch.rand(400, 600) > 0.5
    mask2 = torch.rand(512, 512) > 0.5
    masks_list = [mask1, mask2]
    target_size = (256, 256)
    print(f"Input mask shapes: {[m.shape for m in masks_list]}")
    print(f"Target size: {target_size}")

    try:
        stacked = _stack_masks_with_resize(masks_list, device, target_size=target_size)
        print(f"✓ Successfully stacked! Output shape: {stacked.shape}")
        assert stacked.shape == (2, 256, 256), f"Expected (2, 256, 256), got {stacked.shape}"
        print("✓ Shape is correct!")
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
    return True


if __name__ == "__main__":
    success = test_mask_stacking()
    exit(0 if success else 1)
