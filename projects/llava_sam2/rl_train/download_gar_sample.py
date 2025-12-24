"""
Download a small sample of GAR dataset for testing.
Downloads only 2 arrow files from Fine-Grained-Dataset-Part1.
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from datasets import load_dataset
from huggingface_hub import snapshot_download
import sys

def download_sample():
    """Download a small sample of the GAR dataset."""

    print("=" * 60)
    print("Downloading GAR Dataset Sample")
    print("Using HF Mirror: http://hf-mirror.com")
    print("=" * 60)

    dataset_name = "HaochenWang/Grasp-Any-Region-Dataset"
    cache_dir = "./data/gar_sample_test"

    print(f"\nDataset: {dataset_name}")
    print(f"Cache dir: {cache_dir}")
    print("\nNote: We'll download just a small part for testing...")

    try:
        # Method 1: Try streaming to see structure first
        print("\n[Step 1] Checking dataset structure with streaming...")
        dataset = load_dataset(
            dataset_name,
            split="train",
            streaming=True,
            cache_dir=cache_dir,
        )

        print("Dataset structure:")
        first_sample = next(iter(dataset))
        print(f"Keys: {first_sample.keys()}")
        print(f"Sample caption: {first_sample.get('caption', 'N/A')[:100]}...")

        # Method 2: Download a small subset
        print("\n[Step 2] Downloading a small subset (first few samples)...")
        dataset = load_dataset(
            dataset_name,
            split="train[:100]",  # Only first 100 samples
            cache_dir=cache_dir,
        )

        print(f"\n✓ Successfully downloaded {len(dataset)} samples!")
        print(f"Saved to: {cache_dir}")

        # Print sample info
        print("\n[Step 3] Verifying downloaded data...")
        sample = dataset[0]
        print(f"Sample 0:")
        print(f"  - Image: {type(sample['image'])}")
        print(f"  - Mask: {type(sample['mask'])}")
        print(f"  - Caption: {sample['caption'][:100]}...")

        return cache_dir

    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    cache_dir = download_sample()

    if cache_dir:
        print("\n" + "=" * 60)
        print("✓ Download completed successfully!")
        print("=" * 60)
        print(f"\nData saved to: {cache_dir}")
        print("\nNext: Run the dataloader test:")
        print(f"  python projects/llava_sam2/rl_train/test_dataset_loading.py --local_data_dir {cache_dir}")
        sys.exit(0)
    else:
        print("\n✗ Download failed!")
        sys.exit(1)
