"""
Test loading data from existing HF cache without network access.
"""

import os
import sys
sys.path.insert(0, '/data/xiaoyicheng/Sa2VA')

# Prevent any network access
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

print("=" * 60)
print("Testing Offline Data Loading from HF Cache")
print("=" * 60)

from datasets import load_dataset
import numpy as np
from PIL import Image

def test_cached_data():
    """Try to load from HF cache."""

    cache_dir = "/data/xyc/cache/hub"

    print(f"\nCache directory: {cache_dir}")
    print("Attempting to load from cache (offline mode)...")

    try:
        # Try to load with offline mode
        dataset = load_dataset(
            "HaochenWang/Grasp-Any-Region-Dataset",
            split="train[:10]",
            cache_dir=cache_dir,
            download_mode="reuse_cache_if_exists",  # Don't download, use cache
        )

        print(f"\n✓ Successfully loaded {len(dataset)} samples from cache!")

        # Test first sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Keys: {sample.keys()}")
        print(f"  Image type: {type(sample['image'])}")
        if hasattr(sample['image'], 'size'):
            print(f"  Image size: {sample['image'].size}")
        print(f"  Mask type: {type(sample['mask'])}")

        # Check mask
        mask = sample['mask']
        if isinstance(mask, np.ndarray):
            print(f"  Mask shape: {mask.shape}")
        elif isinstance(mask, list):
            mask_arr = np.array(mask)
            print(f"  Mask shape (as array): {mask_arr.shape}")

        if 'caption' in sample:
            print(f"  Caption: {sample['caption'][:100]}...")

        # Now test with our dataloader
        print("\n" + "=" * 60)
        print("Testing Custom Dataloader")
        print("=" * 60)

        from projects.llava_sam2.rl_train.dataset import collate_fn_sa2va_rl
        from torch.utils.data import Dataset as TorchDataset, DataLoader

        class SimpleWrapper(TorchDataset):
            def __init__(self, hf_dataset):
                self.dataset = hf_dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                sample = self.dataset[idx]

                # Process image
                image = sample['image']
                if not isinstance(image, Image.Image):
                    image = Image.fromarray(np.array(image)).convert('RGB')

                # Process mask
                mask = sample['mask']
                if isinstance(mask, Image.Image):
                    mask = np.array(mask)
                elif isinstance(mask, list):
                    mask = np.array(mask)
                elif not isinstance(mask, np.ndarray):
                    mask = np.array(mask)

                if mask.dtype != np.bool_:
                    mask = mask > 0

                return {
                    'image': image,
                    'mask': mask,
                    'caption': sample.get('caption', ''),
                    'image_id': f'img_{idx}'
                }

        # Wrap dataset
        wrapped = SimpleWrapper(dataset)

        print(f"\nDataset length: {len(wrapped)}")

        # Test single sample
        test_sample = wrapped[0]
        print(f"\nProcessed sample 0:")
        print(f"  Image: {test_sample['image'].size} {test_sample['image'].mode}")
        print(f"  Mask: {test_sample['mask'].shape} {test_sample['mask'].dtype}")
        print(f"  Caption: {test_sample['caption'][:100]}...")

        # Test DataLoader
        print("\nTesting DataLoader...")
        loader = DataLoader(
            wrapped,
            batch_size=2,
            collate_fn=collate_fn_sa2va_rl,
            shuffle=False
        )

        for i, batch in enumerate(loader):
            print(f"  Batch {i}: {len(batch['images'])} samples")
            print(f"    First image: {batch['images'][0].size}")
            print(f"    First mask: {batch['masks'][0].shape}")
            if i >= 0:
                break

        print("\n" + "=" * 60)
        print("✓ All offline tests passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_cached_data()
    sys.exit(0 if success else 1)
