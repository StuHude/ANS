"""
Quick test - load just a few samples from the GAR dataset to verify everything works.
"""

import sys
sys.path.insert(0, '/data/xiaoyicheng/Sa2VA')

print("=" * 60)
print("Quick Test: GAR Dataset Loader")
print("=" * 60)

from projects.llava_sam2.rl_train.dataset_gar import (
    GraspAnyRegionDataset,
    collate_fn_sa2va_rl
)
from torch.utils.data import DataLoader, Subset
import numpy as np

try:
    # Load full dataset
    print("\nLoading dataset...")
    dataset = GraspAnyRegionDataset(
        local_data_dir="/data/xiaoyicheng/Sa2VA/data/GAR",
        parts_to_load=["Fine-Grained-Dataset-Part1"]  # Just Part1 for quick test
    )

    print(f"✓ Total dataset size: {len(dataset)}")

    # Test with just first 5 samples
    print("\n" + "=" * 60)
    print("Testing First 5 Samples")
    print("=" * 60)

    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Image: {sample['image'].size} {sample['image'].mode}")
        print(f"  Mask: {sample['mask'].shape} {sample['mask'].dtype}")
        print(f"  Mask sum: {sample['mask'].sum()}")
        print(f"  Caption: {sample['caption'][:60]}...")
        print(f"  Category: {sample['category']}")

    # Test DataLoader
    print("\n" + "=" * 60)
    print("Testing DataLoader with Batch Size 2")
    print("=" * 60)

    subset = Subset(dataset, range(min(10, len(dataset))))
    loader = DataLoader(
        subset,
        batch_size=2,
        collate_fn=collate_fn_sa2va_rl,
        shuffle=False,
        num_workers=0
    )

    for i, batch in enumerate(loader):
        print(f"\nBatch {i}:")
        print(f"  Images: {len(batch['images'])} samples")
        print(f"  Masks: {len(batch['masks'])} samples")
        print(f"  Captions: {len(batch['captions'])} samples")

        # Show first sample in batch
        print(f"  Sample 0 in batch:")
        print(f"    Image: {batch['images'][0].size}")
        print(f"    Mask: {batch['masks'][0].shape} sum={batch['masks'][0].sum()}")
        print(f"    Caption: {batch['captions'][0][:50]}...")

        if i >= 2:  # Just 3 batches
            break

    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)

    print("\n[Summary]")
    print(f"  ✓ Dataset loaded: {len(dataset)} samples")
    print(f"  ✓ Image loading works")
    print(f"  ✓ RLE mask decoding works")
    print(f"  ✓ Caption extraction works")
    print(f"  ✓ DataLoader batching works")
    print(f"  ✓ Ready for RL training!")

except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
