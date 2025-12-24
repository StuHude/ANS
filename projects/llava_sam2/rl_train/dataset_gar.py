"""
Dataset loader for Grasp-Any-Region (GAR) dataset for RL training.
Reads directly from Arrow files and handles RLE masks.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import numpy as np
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset as HFDataset, concatenate_datasets


def rle_decode(rle_dict, dtype=np.bool_):
    """
    Decode RLE mask to binary mask.

    Args:
        rle_dict: Dict with 'counts' (string) and 'size' (list [H, W])
        dtype: Output dtype

    Returns:
        Binary mask (H, W)
    """
    import pycocotools.mask as mask_util

    # COCO RLE format - decode directly
    # The rle_dict is already in the format pycocotools expects
    mask = mask_util.decode(rle_dict)
    return mask.astype(dtype)


def extract_caption_from_conversations(conversations):
    """
    Extract caption from conversations list.

    Args:
        conversations: List of dicts with 'from' and 'value' keys

    Returns:
        str: Extracted caption
    """
    # Usually the conversation format is:
    # [{"from": "human", "value": question}, {"from": "gpt", "value": answer}]
    # We want the GPT's answer as the caption

    if not conversations:
        return ""

    for conv in conversations:
        if conv.get('from') in ['gpt', 'assistant']:
            return conv.get('value', '')

    # Fallback: return last message
    if conversations:
        return conversations[-1].get('value', '')

    return ""


class GraspAnyRegionDataset(Dataset):
    """
    Dataset for loading Grasp-Any-Region (GAR) dataset from Arrow files.

    Each sample contains an image, an RLE mask, and conversations.
    """
    def __init__(
        self,
        local_data_dir,
        parts_to_load=None,
    ):
        """
        Args:
            local_data_dir: Path to local directory containing Part folders with Arrow files
                          Example: "/data/xiaoyicheng/Sa2VA/data/GAR"
            parts_to_load: List of part folder names to load. If None, loads all.
                          Example: ["Fine-Grained-Dataset-Part1", "Fine-Grained-Dataset-Part2"]
        """
        self.dataset = self._load_from_arrow_files(local_data_dir, parts_to_load)
        print(f"Loaded {len(self.dataset)} samples")

    def _load_from_arrow_files(self, local_data_dir, parts_to_load=None):
        """
        Load dataset directly from Arrow files.

        Args:
            local_data_dir: Base directory containing part folders
            parts_to_load: List of part folder names. If None, auto-detect all parts.

        Returns:
            Concatenated HFDataset from all arrow files
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

        all_tables = []

        for part_name in parts_to_load:
            part_dir = local_data_dir / part_name

            if not part_dir.exists():
                print(f"Warning: Part directory not found: {part_dir}, skipping...")
                continue

            # Find all arrow files in this part
            arrow_files = sorted(part_dir.glob("*.arrow"))

            if not arrow_files:
                print(f"Warning: No arrow files found in {part_dir}")
                continue

            print(f"Loading {len(arrow_files)} arrow files from {part_name}...")

            for arrow_file in arrow_files:
                try:
                    # Read arrow file
                    with pa.memory_map(str(arrow_file), 'r') as source:
                        try:
                            # Try IPC file format
                            reader = pa.ipc.open_file(source)
                            table = reader.read_all()
                        except:
                            # Try stream format
                            source.seek(0)
                            reader = pa.ipc.open_stream(source)
                            table = reader.read_all()

                    all_tables.append(table)
                    print(f"  Loaded {len(table)} rows from {arrow_file.name}")

                except Exception as e:
                    print(f"  Warning: Failed to load {arrow_file.name}: {e}")

        if not all_tables:
            raise ValueError(f"No arrow files loaded from {local_data_dir}")

        # Concatenate all tables
        print(f"Concatenating {len(all_tables)} arrow tables...")
        combined_table = pa.concat_tables(all_tables)

        # Convert to HuggingFace Dataset with custom fingerprint to avoid recursion issues
        import hashlib
        fingerprint = hashlib.md5(f"gar_dataset_{len(combined_table)}".encode()).hexdigest()
        dataset = HFDataset(combined_table, fingerprint=fingerprint)

        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - image: PIL.Image
                - mask: numpy array (H, W) with bool
                - caption: str
                - category: str
                - image_id: str
        """
        sample = self.dataset[idx]

        # Load image from bytes
        image_data = sample['image']
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image = Image.open(io.BytesIO(image_data['bytes'])).convert('RGB')
        else:
            raise ValueError(f"Unexpected image format: {type(image_data)}")

        # Decode RLE mask
        mask_rle = sample['mask_rle']
        try:
            mask = rle_decode(mask_rle, dtype=np.bool_)
        except Exception as e:
            # If RLE decoding fails, create a zero mask
            print(f"Warning: RLE decode failed for sample {idx}: {e}")
            # Use image size for mask
            mask = np.zeros((image.height, image.width), dtype=np.bool_)

        # Extract caption from conversations
        conversations = sample['conversations']
        caption = extract_caption_from_conversations(conversations)

        # Category
        category = sample.get('catagory', '')  # Note: typo in dataset

        result = {
            'image': image,
            'mask': mask,
            'caption': caption,
            'category': category,
            'image_id': f'sample_{idx}',
        }

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
        'categories': [item.get('category', '') for item in batch],
        'image_ids': [item.get('image_id', f'img_{i}') for i, item in enumerate(batch)],
    }
