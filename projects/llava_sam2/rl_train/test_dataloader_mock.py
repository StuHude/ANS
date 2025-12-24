"""
Test dataloader with mock data (without downloading).
This tests the dataloader logic independently of the actual dataset.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import numpy as np
from PIL import Image
from datasets import Dataset
import tempfile

def create_mock_dataset(num_samples=10):
    """Create a mock dataset with synthetic data."""

    print("Creating mock dataset...")

    data = {
        'image': [],
        'mask': [],
        'caption': [],
        'image_id': []
    }

    for i in range(num_samples):
        # Create a simple image
        img = Image.new('RGB', (224, 224), color=(i*25 % 256, 100, 150))

        # Create a simple mask
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask[50:150, 50:150] = 1  # Simple square mask

        # Create caption
        caption = f"This is test image {i} with a square region in the center."

        data['image'].append(img)
        data['mask'].append(mask)
        data['caption'].append(caption)
        data['image_id'].append(f'test_{i:04d}')

    # Create HF dataset
    dataset = Dataset.from_dict(data)

    print(f"Created dataset with {len(dataset)} samples")
    return dataset


def test_with_mock_data():
    """Test the dataloader with mock data."""

    print("=" * 60)
    print("Testing Dataloader with Mock Data")
    print("=" * 60)

    # Import our dataset class
    from projects.llava_sam2.rl_train.dataset import collate_fn_sa2va_rl
    from torch.utils.data import Dataset as TorchDataset

    # Create a simple wrapper
    class MockGARDataset(TorchDataset):
        def __init__(self, hf_dataset):
            self.dataset = hf_dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            sample = self.dataset[idx]

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
                'image': sample['image'],
                'mask': mask,
                'caption': sample['caption'],
                'image_id': sample['image_id']
            }

    # Create mock data
    mock_hf_dataset = create_mock_dataset(num_samples=10)

    # Wrap it
    dataset = MockGARDataset(mock_hf_dataset)

    print(f"\n[Test 1] Dataset length: {len(dataset)}")
    assert len(dataset) == 10, "Dataset length mismatch"
    print("✓ Passed")

    # Test single sample
    print("\n[Test 2] Loading single sample...")
    sample = dataset[0]
    print(f"  - Image: {sample['image'].size} {sample['image'].mode}")
    print(f"  - Mask: {sample['mask'].shape} {sample['mask'].dtype}")
    print(f"  - Caption: {sample['caption']}")
    print(f"  - Image ID: {sample['image_id']}")
    assert isinstance(sample['image'], Image.Image), "Image should be PIL Image"
    assert isinstance(sample['mask'], np.ndarray), "Mask should be numpy array"
    assert sample['mask'].dtype == np.bool_, "Mask should be boolean"
    print("✓ Passed")

    # Test batch collation
    print("\n[Test 3] Testing batch collation...")
    batch_samples = [dataset[i] for i in range(3)]
    batch = collate_fn_sa2va_rl(batch_samples)

    print(f"  - Batch keys: {batch.keys()}")
    print(f"  - Images: {len(batch['images'])} items")
    print(f"  - Masks: {len(batch['masks'])} items")
    print(f"  - Captions: {len(batch['captions'])} items")
    assert len(batch['images']) == 3, "Batch size mismatch"
    assert len(batch['masks']) == 3, "Batch size mismatch"
    assert len(batch['captions']) == 3, "Batch size mismatch"
    print("✓ Passed")

    # Test with DataLoader
    print("\n[Test 4] Testing with PyTorch DataLoader...")
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collate_fn_sa2va_rl,
        shuffle=False
    )

    for i, batch in enumerate(dataloader):
        print(f"  Batch {i}: {len(batch['images'])} samples")
        if i >= 2:  # Test first 3 batches
            break
    print("✓ Passed")

    print("\n" + "=" * 60)
    print("✓ All mock data tests passed!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = test_with_mock_data()

        if success:
            print("\n🎉 Dataloader logic is working correctly!")
            print("\nNext: Download actual GAR dataset samples to test with real data")

        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
