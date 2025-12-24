"""
Gradient monitoring utilities for detecting and debugging NaN/inf issues.

This module provides:
1. Gradient hooks to detect NaN/inf in gradients
2. Gradient norm monitoring
3. Automatic gradient clipping with statistics
"""

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np


class GradientMonitor:
    """
    Monitor gradients during training to detect NaN/inf and track statistics.

    Usage:
        monitor = GradientMonitor(model)
        monitor.attach()

        # During training
        loss.backward()
        stats = monitor.check_gradients()
        if stats['has_nan'] or stats['has_inf']:
            print("Gradient issue detected!")
    """

    def __init__(self, model, verbose=True):
        """
        Args:
            model: PyTorch model to monitor
            verbose: Whether to print warnings
        """
        self.model = model
        self.verbose = verbose
        self.hooks = []
        self.gradient_stats = defaultdict(list)
        self.step_count = 0

    def attach(self):
        """Attach gradient hooks to all parameters."""
        def make_hook(name):
            def hook(grad):
                # Check for NaN/inf
                has_nan = torch.isnan(grad).any().item()
                has_inf = torch.isinf(grad).any().item()

                if has_nan or has_inf:
                    if self.verbose:
                        print(f"⚠ WARNING: Gradient issue in {name}")
                        print(f"  Shape: {grad.shape}")
                        print(f"  Has NaN: {has_nan}, Has Inf: {has_inf}")
                        print(f"  Min: {grad.min().item():.6f}, Max: {grad.max().item():.6f}")

                # Track statistics
                self.gradient_stats[name].append({
                    'norm': grad.norm().item(),
                    'mean': grad.mean().item(),
                    'std': grad.std().item(),
                    'min': grad.min().item(),
                    'max': grad.max().item(),
                    'has_nan': has_nan,
                    'has_inf': has_inf,
                })

                return grad
            return hook

        # Register hooks for all parameters with gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(make_hook(name))
                self.hooks.append(hook)

        if self.verbose:
            print(f"✓ Gradient monitor attached to {len(self.hooks)} parameters")

    def detach(self):
        """Remove all gradient hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def check_gradients(self):
        """
        Check current gradient statistics.

        Returns:
            dict with keys:
                - has_nan: bool
                - has_inf: bool
                - max_grad_norm: float
                - param_with_max_norm: str
                - total_params_with_grad: int
        """
        stats = {
            'has_nan': False,
            'has_inf': False,
            'max_grad_norm': 0.0,
            'param_with_max_norm': None,
            'total_params_with_grad': 0,
            'grad_norms': {},
        }

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                stats['total_params_with_grad'] += 1

                # Check NaN/inf
                if torch.isnan(param.grad).any():
                    stats['has_nan'] = True
                if torch.isinf(param.grad).any():
                    stats['has_inf'] = True

                # Track norm
                grad_norm = param.grad.norm().item()
                stats['grad_norms'][name] = grad_norm

                if grad_norm > stats['max_grad_norm']:
                    stats['max_grad_norm'] = grad_norm
                    stats['param_with_max_norm'] = name

        return stats

    def get_summary(self, top_k=10):
        """
        Get summary statistics of gradients across all steps.

        Args:
            top_k: Number of top parameters to show (by gradient norm)

        Returns:
            dict with summary statistics
        """
        summary = {}

        for name, stats_list in self.gradient_stats.items():
            if not stats_list:
                continue

            norms = [s['norm'] for s in stats_list]
            summary[name] = {
                'mean_norm': np.mean(norms),
                'max_norm': np.max(norms),
                'min_norm': np.min(norms),
                'std_norm': np.std(norms),
                'nan_count': sum(s['has_nan'] for s in stats_list),
                'inf_count': sum(s['has_inf'] for s in stats_list),
            }

        # Sort by mean norm and get top k
        sorted_params = sorted(summary.items(), key=lambda x: x[1]['mean_norm'], reverse=True)

        return {
            'all_params': summary,
            'top_params': dict(sorted_params[:top_k]),
            'total_params': len(summary),
        }

    def reset_stats(self):
        """Reset all collected statistics."""
        self.gradient_stats.clear()
        self.step_count = 0


class GradientClipper:
    """
    Enhanced gradient clipper with monitoring and statistics.

    This is a wrapper around torch.nn.utils.clip_grad_norm_ that:
    1. Tracks clipping statistics
    2. Detects when clipping is triggered
    3. Provides warnings if gradients are consistently large
    """

    def __init__(self, max_norm=1.0, norm_type=2.0, verbose=True):
        """
        Args:
            max_norm: Maximum gradient norm
            norm_type: Type of norm (2 for L2 norm)
            verbose: Whether to print warnings
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.verbose = verbose

        self.clip_count = 0
        self.total_count = 0
        self.norm_history = []

    def clip(self, parameters):
        """
        Clip gradients and return total norm.

        Args:
            parameters: Model parameters (or iterable of parameters)

        Returns:
            total_norm: Total gradient norm before clipping
        """
        self.total_count += 1

        # Compute total norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            self.max_norm,
            norm_type=self.norm_type
        )

        total_norm_value = total_norm.item()
        self.norm_history.append(total_norm_value)

        # Check if clipping was triggered
        if total_norm_value > self.max_norm:
            self.clip_count += 1

            if self.verbose and self.clip_count <= 10:  # Print first 10 times
                print(f"⚠ Gradient clipping triggered (#{self.clip_count}): "
                      f"norm={total_norm_value:.2f} -> {self.max_norm}")

        # Warn if clipping happens too frequently
        if self.total_count > 100 and self.clip_count / self.total_count > 0.5:
            if self.verbose and self.total_count % 100 == 0:
                print(f"⚠ WARNING: Gradient clipping triggered {self.clip_count}/{self.total_count} "
                      f"({100*self.clip_count/self.total_count:.1f}%) times. "
                      f"Consider reducing learning rate or checking for numerical issues.")

        return total_norm_value

    def get_stats(self):
        """Get clipping statistics."""
        if not self.norm_history:
            return {}

        return {
            'clip_count': self.clip_count,
            'total_count': self.total_count,
            'clip_rate': self.clip_count / max(self.total_count, 1),
            'mean_norm': np.mean(self.norm_history),
            'max_norm': np.max(self.norm_history),
            'min_norm': np.min(self.norm_history),
            'std_norm': np.std(self.norm_history),
        }

    def reset_stats(self):
        """Reset statistics."""
        self.clip_count = 0
        self.total_count = 0
        self.norm_history = []
