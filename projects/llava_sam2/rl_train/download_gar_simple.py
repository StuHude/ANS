"""
Simple script to download GAR dataset using HF mirror.
Downloads a small sample for testing.
"""

import os

# MUST set this BEFORE importing from datasets!
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print("=" * 60)
print("Downloading GAR Dataset Sample")
print(f"Using HF Mirror: {os.environ['HF_ENDPOINT']}")
print("=" * 60)

from datasets import load_dataset
import sys

def download_sample():
    """Download a small sample."""

    dataset_name = "HaochenWang/Grasp-Any-Region-Dataset"
    cache_dir = "./data/gar_test_sample"

    print(f"\nDataset: {dataset_name}")
    print(f"Cache dir: {cache_dir}")
    print("\nDownloading first 50 samples for testing...")

    try:
        # Download first 50 samples only
        dataset = load_dataset(
            dataset_name,
            split="train[:50]",
            cache_dir=cache_dir,
        )

        print(f"\n✓ Successfully downloaded {len(dataset)} samples!")

        # Test sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  - Keys: {sample.keys()}")
        print(f"  - Image type: {type(sample['image'])}")
        print(f"  - Mask type: {type(sample['mask'])}")
        print(f"  - Caption: {str(sample['caption'])[:100]}...")

        return cache_dir

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = download_sample()
    sys.exit(0 if result else 1)
