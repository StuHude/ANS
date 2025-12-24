"""
Download a small GAR dataset sample and test the dataloader.
Uses HF mirror and works offline after download.
"""

import os
import sys

# CRITICAL: Set mirror BEFORE any imports
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("GAR Dataset Download and Test")
print(f"Using HF Mirror: {os.environ['HF_ENDPOINT']}")
print("=" * 60)

from datasets import load_dataset, Dataset
from PIL import Image
import numpy as np

def download_sample(num_samples=50):
    """Download a small sample using HF mirror."""

    dataset_name = "HaochenWang/Grasp-Any-Region-Dataset"
    cache_dir = "./data/gar_test_sample"

    print(f"\nDataset: {dataset_name}")
    print(f"Cache dir: {cache_dir}")
    print(f"Downloading first {num_samples} samples...")

    try:
        dataset = load_dataset(
            dataset_name,
            split=f"train[:{num_samples}]",
            cache_dir=cache_dir,
        )

        print(f"\n✓ Successfully downloaded {len(dataset)} samples!")
        return dataset, cache_dir

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_dataloader(dataset):
    """Test the dataloader with downloaded data."""

    print("\n" + "=" * 60)
    print("Testing Dataloader")
    print("=" * 60)

    if dataset is None:
        print("No dataset to test!")
        return False

    try:
        # Test sample structure
        print(f"\nDataset length: {len(dataset)}")

        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Keys: {sample.keys()}")
        print(f"  Image type: {type(sample['image'])}")
        print(f"  Image size: {sample['image'].size if hasattr(sample['image'], 'size') else 'N/A'}")
        print(f"  Mask type: {type(sample['mask'])}")

        # Check mask format
        mask = sample['mask']
        if isinstance(mask, np.ndarray):
            print(f"  Mask shape: {mask.shape}")
            print(f"  Mask dtype: {mask.dtype}")
        elif isinstance(mask, list):
            mask_arr = np.array(mask)
            print(f"  Mask shape (as array): {mask_arr.shape}")
            print(f"  Mask dtype (as array): {mask_arr.dtype}")

        print(f"  Caption: {str(sample.get('caption', 'N/A'))[:100]}...")

        # Now test with our custom dataset wrapper
        print("\n" + "=" * 60)
        print("Testing Custom Dataset Wrapper")
        print("=" * 60)

        sys.path.insert(0, '/data/xiaoyicheng/Sa2VA')
        from projects.llava_sam2.rl_train.dataset import GraspAnyRegionDataset, collate_fn_sa2va_rl
        from torch.utils.data import DataLoader

        # Create a mock local dataset by saving and loading
        print("\nCreating test dataset from downloaded data...")

        # Save to disk for local loading test
        save_path = "./data/gar_test_local"
        dataset.save_to_disk(save_path)
        print(f"Saved dataset to: {save_path}")

        # Test loading from local
        print("\nTesting local loading...")
        local_dataset = GraspAnyRegionDataset(
            local_data_dir=save_path,
            parts_to_load=None  # Auto-detect
        )

        print(f"✓ Loaded {len(local_dataset)} samples from local")

        # Test single sample
        sample = local_dataset[0]
        print(f"\nSample 0:")
        print(f"  Image: {sample['image'].size} {sample['image'].mode}")
        print(f"  Mask: {sample['mask'].shape} {sample['mask'].dtype}")
        print(f"  Caption: {sample['caption'][:100]}...")

        # Test dataloader
        print("\nTesting PyTorch DataLoader...")
        dataloader = DataLoader(
            local_dataset,
            batch_size=2,
            collate_fn=collate_fn_sa2va_rl,
            shuffle=False
        )

        for i, batch in enumerate(dataloader):
            print(f"  Batch {i}: {len(batch['images'])} samples")
            if i >= 1:
                break

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Download
    dataset, cache_dir = download_sample(num_samples=50)

    if dataset is None:
        sys.exit(1)

    # Test
    success = test_dataloader(dataset)

    sys.exit(0 if success else 1)
