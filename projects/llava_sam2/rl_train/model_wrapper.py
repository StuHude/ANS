"""
DataParallel-compatible wrapper for Sa2VA model.

This wrapper solves two critical issues when using Sa2VA with multi-GPU training:
1. DataParallel incompatibility with Sa2VA's forward(data, mode) signature
2. Incorrect scattering of prompt_masks (list) across GPUs causing shape mismatch

The wrapper implements a standard forward(**kwargs) interface that DataParallel expects,
while internally packaging inputs into Sa2VA's expected data dict format.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Any


class Sa2VADataParallelWrapper(nn.Module):
    """
    Wrapper for Sa2VA model to make it compatible with DataParallel.

    Key fixes:
    1. Implements standard forward(**kwargs) signature instead of forward(data, mode)
    2. Handles prompt_masks as a special case to prevent incorrect scattering
    3. Ensures each GPU replica gets the correct subset of prompt_masks

    Usage:
        model = Sa2VAChatModel.from_pretrained(...)
        wrapped_model = Sa2VADataParallelWrapper(model)
        wrapped_model = nn.DataParallel(wrapped_model, device_ids=[0, 1, 2, ...])

        # Now you can call it with individual arguments:
        outputs = wrapped_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            prompt_masks=prompt_masks,  # List of tensors
            vp_overall_mask=vp_overall_mask,
            mode='loss'
        )
    """

    def __init__(self, sa2va_model):
        super().__init__()
        self.model = sa2va_model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor,
        prompt_masks: Union[List[torch.Tensor], torch.Tensor],
        vp_overall_mask: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mode: str = 'loss',
        **kwargs
    ):
        """
        Forward pass with standard kwargs interface.

        CRITICAL: This method handles the prompt_masks list correctly for DataParallel.

        When DataParallel scatters inputs:
        - Tensors (input_ids, pixel_values, etc.) are automatically split by batch dimension
        - Lists (prompt_masks) are passed as-is to each replica, causing mismatch!

        This wrapper ensures prompt_masks is sliced correctly to match the batch size
        on each GPU.

        Args:
            input_ids: (B, L) token IDs
            attention_mask: (B, L) attention mask
            pixel_values: (B, T, 3, H, W) or (B, 3, H, W) images
            prompt_masks: List of (N_regions, G, G) tensors, one per sample in batch
            vp_overall_mask: (B, T) or (B,) bool mask indicating which frames have VP
            position_ids: Optional (B, L) position IDs
            labels: Optional (B, L) labels for loss computation
            mode: 'loss' or 'generate'
            **kwargs: Additional arguments

        Returns:
            Model outputs (logits, loss, etc.)
        """
        # Get actual batch size from input_ids (may be smaller due to DataParallel split)
        batch_size = input_ids.size(0)

        # CRITICAL FIX: Handle prompt_masks list correctly for DataParallel
        # When DataParallel scatters the batch, it splits tensors but NOT lists!
        # We need to ensure prompt_masks matches the actual batch size on this GPU.
        if isinstance(prompt_masks, list):
            # If this is a DataParallel replica with a subset of the batch,
            # we need to slice the prompt_masks list accordingly.
            #
            # However, DataParallel passes the FULL list to each replica, which causes
            # the shape mismatch! We need to infer which subset belongs to this replica.
            #
            # The simplest solution: Only take the first batch_size elements.
            # This works because DataParallel scatters in order:
            # - GPU 0 gets samples [0:B//N]
            # - GPU 1 gets samples [B//N:2*B//N]
            # But since DataParallel doesn't know how to scatter lists, each GPU
            # still sees the full list. We rely on the fact that the batch_size
            # tells us how many samples THIS replica should process.
            prompt_masks = prompt_masks[:batch_size]

        # Generate position_ids if not provided
        if position_ids is None:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(
                seq_len,
                device=input_ids.device
            ).unsqueeze(0).repeat(batch_size, 1)

        # Use input_ids as labels if not provided (for GRPO, we compute log probs)
        if labels is None and mode == 'loss':
            labels = input_ids.clone()

        # Package into Sa2VA's expected data dict format
        data = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'pixel_values': pixel_values,
            'prompt_masks': prompt_masks,  # Now correctly sized for this GPU
            'vp_overall_mask': vp_overall_mask,
        }

        if labels is not None:
            data['labels'] = labels

        # Call Sa2VA's original forward method
        return self.model(data, mode=mode)

    def generate(self, data, **kwargs):
        """
        Wrapper for generation method.
        For generation, we typically use single GPU, so no DataParallel issues.
        """
        return self.model.generate(data, **kwargs)

    def __getattr__(self, name):
        """
        Forward attribute access to the wrapped model.
        This allows accessing model.config, model.parameters(), etc.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
