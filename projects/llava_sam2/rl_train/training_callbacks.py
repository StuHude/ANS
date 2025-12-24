"""
Training callbacks for monitoring and debugging Sa2VA RL training.

These callbacks help detect and diagnose NaN/inf issues during training.
"""

import torch
import numpy as np
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl


class GradientMonitorCallback(TrainerCallback):
    """
    Callback to monitor gradients during training.

    This callback:
    1. Checks for NaN/inf in gradients after backward pass
    2. Tracks gradient norms
    3. Logs statistics periodically
    4. Can halt training if serious issues are detected
    """

    def __init__(self, check_every_n_steps=1, log_every_n_steps=10, halt_on_nan=False):
        """
        Args:
            check_every_n_steps: How often to check gradients
            log_every_n_steps: How often to log statistics
            halt_on_nan: Whether to stop training if NaN is detected
        """
        self.check_every_n_steps = check_every_n_steps
        self.log_every_n_steps = log_every_n_steps
        self.halt_on_nan = halt_on_nan

        self.step_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.grad_norm_history = []

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called after each training step (after backward, before optimizer step)."""
        self.step_count += 1

        # Only check every N steps
        if self.step_count % self.check_every_n_steps != 0:
            return control

        model = kwargs.get('model')
        if model is None:
            return control

        # Check gradients
        has_nan = False
        has_inf = False
        grad_norms = []
        params_with_issues = []

        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check for NaN/inf
                if torch.isnan(param.grad).any():
                    has_nan = True
                    params_with_issues.append(f"{name} (NaN)")

                if torch.isinf(param.grad).any():
                    has_inf = True
                    params_with_issues.append(f"{name} (Inf)")

                # Track norm
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)

        # Update statistics
        if has_nan:
            self.nan_count += 1
        if has_inf:
            self.inf_count += 1

        # Compute total gradient norm
        total_grad_norm = np.sqrt(sum(n**2 for n in grad_norms)) if grad_norms else 0.0
        self.grad_norm_history.append(total_grad_norm)

        # Log issues
        if has_nan or has_inf:
            print(f"\n{'='*70}")
            print(f"⚠ GRADIENT ISSUE DETECTED at step {self.step_count}")
            print(f"{'='*70}")
            print(f"  NaN detected: {has_nan} (total count: {self.nan_count})")
            print(f"  Inf detected: {has_inf} (total count: {self.inf_count})")
            print(f"  Total gradient norm: {total_grad_norm:.6f}")
            print(f"\nParameters with issues:")
            for param_name in params_with_issues[:10]:  # Show first 10
                print(f"  - {param_name}")
            if len(params_with_issues) > 10:
                print(f"  ... and {len(params_with_issues) - 10} more")
            print(f"{'='*70}\n")

            # Optionally halt training
            if self.halt_on_nan and has_nan:
                print("⛔ Halting training due to NaN in gradients (halt_on_nan=True)")
                control.should_training_stop = True

        # Log statistics periodically
        if self.step_count % self.log_every_n_steps == 0:
            if len(self.grad_norm_history) > 0:
                recent_norms = self.grad_norm_history[-self.log_every_n_steps:]
                print(f"\n[Step {self.step_count}] Gradient Statistics:")
                print(f"  Mean grad norm: {np.mean(recent_norms):.6f}")
                print(f"  Max grad norm: {np.max(recent_norms):.6f}")
                print(f"  Min grad norm: {np.min(recent_norms):.6f}")
                print(f"  NaN count: {self.nan_count}, Inf count: {self.inf_count}")

        return control

    def get_stats(self):
        """Get summary statistics."""
        return {
            'total_steps': self.step_count,
            'nan_count': self.nan_count,
            'inf_count': self.inf_count,
            'mean_grad_norm': np.mean(self.grad_norm_history) if self.grad_norm_history else 0.0,
            'max_grad_norm': np.max(self.grad_norm_history) if self.grad_norm_history else 0.0,
        }


class ActivationMonitorCallback(TrainerCallback):
    """
    Callback to monitor activations during forward pass.

    This helps detect numerical issues before they cause NaN gradients.
    """

    def __init__(self, check_every_n_steps=10):
        """
        Args:
            check_every_n_steps: How often to check activations
        """
        self.check_every_n_steps = check_every_n_steps
        self.step_count = 0
        self.hooks = []
        self.activation_stats = {}

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of each training step."""
        self.step_count += 1

        # Only check every N steps
        if self.step_count % self.check_every_n_steps != 0:
            return control

        model = kwargs.get('model')
        if model is None:
            return control

        # Register hooks to monitor activations
        self._register_hooks(model)

        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        # Remove hooks
        self._remove_hooks()

        # Report any issues found
        if self.activation_stats:
            has_issues = any(stats.get('has_nan') or stats.get('has_inf')
                           for stats in self.activation_stats.values())

            if has_issues:
                print(f"\n⚠ Activation issues detected at step {self.step_count}:")
                for name, stats in self.activation_stats.items():
                    if stats.get('has_nan') or stats.get('has_inf'):
                        print(f"  {name}: NaN={stats['has_nan']}, Inf={stats['has_inf']}, "
                              f"Max={stats['max']:.6f}, Min={stats['min']:.6f}")

        self.activation_stats = {}
        return control

    def _register_hooks(self, model):
        """Register forward hooks to monitor activations."""
        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_stats[name] = {
                        'has_nan': torch.isnan(output).any().item(),
                        'has_inf': torch.isinf(output).any().item(),
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item(),
                    }
            return hook

        # Register hooks for key modules
        # For Sa2VA, we want to monitor LLM layers and projection layers
        for name, module in model.named_modules():
            # Monitor only specific layer types to avoid too many hooks
            if any(layer_type in name for layer_type in ['mlp', 'self_attn', 'mlp1', 'text_hidden_fcs']):
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove all forward hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
