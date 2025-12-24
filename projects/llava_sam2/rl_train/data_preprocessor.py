"""
Data preprocessor for Sa2VA RL training.
Converts raw data (PIL Image, numpy mask, caption) to model input format.

Reference: projects/llava_sam2/datasets/describe_anything_referring_dataset.py
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np
from typing import Dict, Any, List

# Constants from describe_anything_referring_dataset.py
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.229)
INTERNVL_IMAGE_SIZE = 448
PATCH_SIZE = 14
DOWNSAMPLE_RATIO = 0.5

# Token grid size
GRID_SIZE = int((INTERNVL_IMAGE_SIZE // PATCH_SIZE) * DOWNSAMPLE_RATIO)  # 16
IMG_TOKENS_PER_FRAME = GRID_SIZE * GRID_SIZE  # 256

# Special tokens
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
VP_START_TOKEN = '<vp>'
VP_END_TOKEN = '</vp>'


class Sa2VADataPreprocessor:
    """
    Preprocessor for Sa2VA model input.
    Converts raw data to the format expected by Sa2VA model.
    """

    def __init__(self, image_size=INTERNVL_IMAGE_SIZE):
        self.image_size = image_size
        self.grid_size = GRID_SIZE

        # Image transform
        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess PIL Image to tensor.

        Args:
            image: PIL Image (RGB)

        Returns:
            Tensor of shape (3, H, W)
        """
        if not isinstance(image, Image.Image):
            raise ValueError(f"Expected PIL Image, got {type(image)}")

        if image.mode != 'RGB':
            image = image.convert('RGB')

        return self.image_transform(image)

    def preprocess_mask(self, mask: np.ndarray, image_tensor: torch.Tensor) -> tuple:
        """
        Preprocess mask to token grid format.

        Args:
            mask: numpy array (H, W) with bool or uint8
            image_tensor: preprocessed image tensor (3, H, W)

        Returns:
            prompt_masks: (1, G, G) uint8 binary mask on token grid
            region_pixels: list with single int, number of tokens in the region
        """
        # Convert to torch tensor
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
        else:
            mask_tensor = torch.tensor(mask, dtype=torch.float32)

        # Add batch dim if needed: (H, W) -> (1, H, W)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        elif mask_tensor.ndim == 3 and mask_tensor.shape[0] != 1:
            # If (C, H, W), take first channel
            mask_tensor = mask_tensor[0:1]

        # Resize to match image size
        if mask_tensor.shape[-2:] != image_tensor.shape[-2:]:
            mask_resizer = transforms.Resize(
                image_tensor.shape[-2:],
                interpolation=InterpolationMode.NEAREST
            )
            mask_tensor = mask_resizer(mask_tensor)

        # Aggregate to token grid G×G (16×16)
        pooled = F.adaptive_avg_pool2d(mask_tensor, (self.grid_size, self.grid_size))

        # Binarize
        prompt_masks = (pooled > 0.5).to(torch.uint8)  # (1, G, G)

        # Count number of tokens in region
        region_pixels = [int(prompt_masks[0].sum().item())]

        return prompt_masks, region_pixels

    def build_mask_caption_prompt(self, region_pixels: List[int], instruction: str) -> str:
        """
        Build prompt for mask->caption task.

        Args:
            region_pixels: list of K values (number of tokens per region)
            instruction: instruction template

        Returns:
            Formatted prompt string
        """
        n_regions = len(region_pixels)
        start_region = f"<image> There are {n_regions} part regions in the picture: "

        parts = []
        for i in range(n_regions):
            K = int(region_pixels[i])
            parts.append(f"region{i+1}{VP_START_TOKEN}{IMG_CONTEXT_TOKEN * K}{VP_END_TOKEN}")

        start_region += (", ".join(parts) + ".\n")
        return start_region + instruction

    def prepare_for_model(
        self,
        image: Image.Image,
        mask: np.ndarray,
        caption: str = None,
        instruction: str = "Please generate a detailed description for the given image region.",
        task: str = "mask_to_caption"
    ) -> Dict[str, Any]:
        """
        Prepare complete model input.

        Args:
            image: PIL Image
            mask: numpy array mask
            caption: ground truth caption (optional, for caption->mask task)
            instruction: instruction template
            task: "mask_to_caption" or "caption_to_mask"

        Returns:
            dict with:
                - pixel_values: (1, 3, H, W)
                - prompt_masks: (1, G, G)
                - vp_overall_mask: (1,) bool
                - prompt_text: str
                - gt_caption: str (if provided)
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)  # (3, H, W)
        pixel_values = image_tensor.unsqueeze(0)  # (1, 3, H, W)

        # Preprocess mask
        prompt_masks, region_pixels = self.preprocess_mask(mask, image_tensor)

        # Visual prompt mask (single frame -> True)
        n_tiles = pixel_values.shape[0]
        vp_overall_mask = torch.tensor([True] + [False] * (n_tiles - 1), dtype=torch.bool)

        # Build prompt
        if task == "mask_to_caption":
            prompt_text = self.build_mask_caption_prompt(region_pixels, instruction)
        elif task == "caption_to_mask":
            # For caption->mask, the caption is the input
            prompt_text = f"<image> {caption}\n{instruction}"
        else:
            raise ValueError(f"Unknown task: {task}")

        result = {
            'pixel_values': pixel_values,        # (1, 3, H, W)
            'prompt_masks': prompt_masks,        # (1, G, G)
            'vp_overall_mask': vp_overall_mask,  # (1,)
            'prompt_text': prompt_text,
            'region_pixels': region_pixels,
        }

        if caption is not None:
            result['gt_caption'] = caption

        return result


def collate_preprocessed_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate preprocessed samples into a batch.

    Args:
        batch: list of dicts from prepare_for_model

    Returns:
        Batched dict
    """
    pixel_values = torch.cat([item['pixel_values'] for item in batch], dim=0)
    prompt_masks_list = [item['prompt_masks'] for item in batch]
    vp_overall_masks = torch.stack([item['vp_overall_mask'] for item in batch], dim=0)
    prompt_texts = [item['prompt_text'] for item in batch]

    result = {
        'pixel_values': pixel_values,
        'prompt_masks': prompt_masks_list,  # list of (N_regions, G, G)
        'vp_overall_mask': vp_overall_masks,
        'prompt_texts': prompt_texts,
    }

    # Optional fields
    if 'gt_caption' in batch[0]:
        result['gt_captions'] = [item['gt_caption'] for item in batch]

    if 'region_pixels' in batch[0]:
        result['region_pixels'] = [item['region_pixels'] for item in batch]

    return result
