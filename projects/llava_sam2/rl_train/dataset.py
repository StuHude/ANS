"""
Dataset loader for Grasp-Any-Region-Dataset for RL training
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from PIL import Image
import io
import numpy as np
import os
from pathlib import Path

class GraspAnyRegionDataset(Dataset):
    """
    Dataset for loading Grasp-Any-Region-Dataset.
    Supports loading from:
    1. HuggingFace Hub (online)
    2. Local Arrow files (offline)

    Each sample contains an image, a mask, and a caption.
    """
    def __init__(
        self,
        dataset_name="HaochenWang/Grasp-Any-Region-Dataset",
        split="train",
        cache_dir="./data/grasp_any_region_cache",
        local_data_dir=None,
        parts_to_load=None,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (used when local_data_dir is None)
            split: train/validation/test
            cache_dir: cache directory for downloaded dataset
            local_data_dir: Path to local directory containing Arrow files.
                          If provided, will load from local instead of HuggingFace.
                          Example: "/path/to/grasp_dataset"
            parts_to_load: List of part folder names to load. If None, loads all.
                          Example: ["Fine-Grained-Dataset-Part1", "Fine-Grained-Dataset-Part2"]
        """
        if local_data_dir is not None:
            # Load from local Arrow files
            self.dataset = self._load_from_local(local_data_dir, parts_to_load)
        else:
            # Load from HuggingFace Hub
            print(f"Loading dataset {dataset_name} split {split} from HuggingFace...")
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                cache_dir=cache_dir,
            )

        print(f"Loaded {len(self.dataset)} samples")

    def _load_from_local(self, local_data_dir, parts_to_load=None):
        """
        Load dataset from local Arrow files.

        Args:
            local_data_dir: Base directory containing part folders
            parts_to_load: List of part folder names. If None, auto-detect all parts.

        Returns:
            Concatenated dataset from all parts
        """
        local_data_dir = Path(local_data_dir)

        if not local_data_dir.exists():
            raise ValueError(f"Local data directory does not exist: {local_data_dir}")

        # Auto-detect parts if not specified
        if parts_to_load is None:
            parts_to_load = []
            for item in sorted(local_data_dir.iterdir()):
                if item.is_dir() and "Fine-Grained-Dataset-Part" in item.name:
                    parts_to_load.append(item.name)

            if not parts_to_load:
                raise ValueError(f"No Fine-Grained-Dataset-Part* folders found in {local_data_dir}")

        print(f"Loading from local directory: {local_data_dir}")
        print(f"Parts to load: {parts_to_load}")

        datasets = []
        for part_name in parts_to_load:
            part_dir = local_data_dir / part_name

            if not part_dir.exists():
                print(f"Warning: Part directory not found: {part_dir}, skipping...")
                continue

            # Try to load from the part directory
            # Arrow files are usually in subdirectories like "train", "test", etc.
            # or directly in the part folder
            try:
                # Try loading directly
                part_dataset = load_from_disk(str(part_dir))
                datasets.append(part_dataset)
                print(f"Loaded {len(part_dataset)} samples from {part_name}")
            except Exception as e1:
                # Try loading from train subdirectory
                train_dir = part_dir / "train"
                if train_dir.exists():
                    try:
                        part_dataset = load_from_disk(str(train_dir))
                        datasets.append(part_dataset)
                        print(f"Loaded {len(part_dataset)} samples from {part_name}/train")
                    except Exception as e2:
                        print(f"Warning: Failed to load {part_name}: {e1}, {e2}")
                else:
                    print(f"Warning: Failed to load {part_name}: {e1}")

        if not datasets:
            raise ValueError(f"No datasets loaded from {local_data_dir}")

        # Concatenate all parts
        if len(datasets) == 1:
            combined_dataset = datasets[0]
        else:
            print(f"Concatenating {len(datasets)} parts...")
            combined_dataset = concatenate_datasets(datasets)

        return combined_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - image: PIL.Image
                - mask: numpy array (H, W) with bool/uint8
                - caption: str
                - image_id: str (optional)
        """
        sample = self.dataset[idx]

        # Load image
        image = sample['image']
        if isinstance(image, dict) and 'bytes' in image:
            # If image is stored as bytes
            image = Image.open(io.BytesIO(image['bytes'])).convert('RGB')
        elif not isinstance(image, Image.Image):
            # Try converting to PIL Image
            image = Image.fromarray(np.array(image)).convert('RGB')

        # Load mask
        mask = sample['mask']
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        elif isinstance(mask, dict) and 'bytes' in mask:
            mask = Image.open(io.BytesIO(mask['bytes']))
            mask = np.array(mask)
        elif isinstance(mask, list):
            # HuggingFace Dataset might convert numpy arrays to lists
            mask = np.array(mask)
        elif not isinstance(mask, np.ndarray):
            # Try converting to numpy array
            mask = np.array(mask)

        # Ensure mask is boolean
        if mask.dtype != np.bool_:
            mask = mask > 0

        # Load caption
        caption = sample['caption']

        result = {
            'image': image,
            'mask': mask,
            'caption': caption,
        }

        # Add image_id if available
        if 'image_id' in sample:
            result['image_id'] = sample['image_id']

        return result


def collate_fn_sa2va_rl(batch):
    """
    Collate function for Sa2VA RL training.
    We don't do preprocessing here - it will be done in the model forward.
    """
    return {
        'images': [item['image'] for item in batch],
        'masks': [item['mask'] for item in batch],
        'captions': [item['caption'] for item in batch],
        'image_ids': [item.get('image_id', f'img_{i}') for i, item in enumerate(batch)],
    }
