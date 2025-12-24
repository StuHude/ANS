"""
Exponential Moving Average (EMA) model wrapper for Sa2VA RL training.
Model 2 is EMA-updated from Model 1.
"""

import torch
import torch.nn as nn
from copy import deepcopy


class EMAModel(nn.Module):
    """
    Exponential Moving Average wrapper for a model.

    Usage:
        student_model = Sa2VAChatModel(...)
        ema_model = EMAModel(student_model, decay=0.999)

        # Training loop
        for batch in dataloader:
            loss = train_step(student_model, batch)
            loss.backward()
            optimizer.step()

            # Update EMA model
            ema_model.update(student_model)
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model: The student model to track
            decay: EMA decay rate (0 < decay < 1)
        """
        super().__init__()
        self.decay = decay
        self.model = deepcopy(model)
        self.model.eval()

        # Disable gradients for EMA model
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update(self, student_model):
        """
        Update EMA parameters using the student model.

        EMA formula: ema_param = decay * ema_param + (1 - decay) * student_param

        Args:
            student_model: The current student model
        """
        student_params = dict(student_model.named_parameters())
        ema_params = dict(self.model.named_parameters())

        for name, ema_param in ema_params.items():
            if name in student_params:
                student_param = student_params[name]
                # Update EMA parameters
                ema_param.data.mul_(self.decay).add_(
                    student_param.data, alpha=1 - self.decay
                )

    def forward(self, *args, **kwargs):
        """Forward pass through the EMA model."""
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def predict_forward(self, *args, **kwargs):
        """Prediction using the EMA model."""
        return self.model.predict_forward(*args, **kwargs)

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
