"""
Quick test script to verify dataset loading from local Arrow files.

Usage:
    python projects/llava_sam2/rl_train/test_dataset_loading.py \
        --local_data_dir /path/to/your/grasp_dataset
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from projects.llava_sam2.rl_train import GraspAnyRegionDataset


def test_dataset(local_data_dir=None, parts_to_load=None):
    """Test loading dataset from local Arrow files."""

    print("=" * 60)
    print("Testing Dataset Loading")
    print("=" * 60)

    if local_data_dir:
        print(f"\nLoading from local directory: {local_data_dir}")
        if parts_to_load:
            print(f"Parts to load: {parts_to_load}")
        else:
            print("Parts to load: All available parts (auto-detect)")
    else:
        print("\nLoading from HuggingFace Hub (online)")

    # Load dataset
    try:
        dataset = GraspAnyRegionDataset(
            local_data_dir=local_data_dir,
            parts_to_load=parts_to_load,
        )
        print(f"\n✓ Successfully loaded dataset!")
        print(f"Total samples: {len(dataset)}")

        # Test loading a few samples
        print("\n" + "=" * 60)
        print("Testing sample loading...")
        print("=" * 60)

        num_test_samples = min(3, len(dataset))
        for i in range(num_test_samples):
            print(f"\nSample {i}:")
            sample = dataset[i]

            print(f"  - Image: {sample['image'].size} {sample['image'].mode}")
            print(f"  - Mask: {sample['mask'].shape} {sample['mask'].dtype}")
            print(f"  - Caption: {sample['caption'][:100]}...")  # First 100 chars

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test dataset loading")

    parser.add_argument(
        "--local_data_dir",
        type=str,
        default=None,
        help="Path to local directory containing Fine-Grained-Dataset-Part* folders"
    )
    parser.add_argument(
        "--parts_to_load",
        type=str,
        nargs="+",
        default=None,
        help="Specific parts to load (e.g., Fine-Grained-Dataset-Part1 Part2)"
    )

    args = parser.parse_args()

    # Auto-detect common paths if not provided
    if args.local_data_dir is None:
        common_paths = [
            "/data/xyc/cache/hub/datasets--HaochenWang--Grasp-Any-Region-Dataset/snapshots/9422475719852204c04762f299967c3a4ca58287",
            "./data/grasp_any_region_cache",
        ]

        print("No --local_data_dir provided, trying common paths...")
        for path in common_paths:
            if os.path.exists(path):
                print(f"Found: {path}")
                args.local_data_dir = path
                break

        if args.local_data_dir is None:
            print("\nNo local data directory found.")
            print("Will attempt to load from HuggingFace Hub (requires internet).")
            print("\nTo test local loading, use:")
            print("  python test_dataset_loading.py --local_data_dir /path/to/your/data")

    success = test_dataset(args.local_data_dir, args.parts_to_load)

    if success:
        print("\n🎉 Dataset is ready for training!")
        print("\nNext steps:")
        print("  1. Review the configuration in run_rl_train.sh")
        print("  2. Update LOCAL_DATA_DIR to your actual path")
        print("  3. Run: bash projects/llava_sam2/rl_train/run_rl_train.sh")
    else:
        print("\n❌ Dataset loading failed. Please check:")
        print("  1. The local_data_dir path is correct")
        print("  2. The directory contains Fine-Grained-Dataset-Part* folders")
        print("  3. The Arrow files are present in the part folders")
        print("\nSee LOCAL_DATA_GUIDE.md for detailed troubleshooting.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
