"""
Standard LoRA - Wraps existing lora_unet.py implementation.

Applies LoRA to attention Q, K, V, Out projections in the UNet.
This is the default LoRA used for fine-tuning InstructPix2Pix.
"""

import torch.nn as nn
from typing import List, Dict, Any

from .base_lora import BaseLoRA

# Import from existing lora_unet.py (preserved as-is)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora_unet import (
    LoRALinear,
    apply_lora_to_unet,
    mark_only_lora_as_trainable,
    get_lora_parameters,
    count_parameters as count_params_legacy,
    _replace_linear_with_lora,
)


class StandardLoRA(BaseLoRA):
    """
    Standard LoRA implementation.

    Applies low-rank adaptation to attention layers (Q, K, V, Out projections).
    Wraps the existing lora_unet.py functions.

    Args:
        rank: LoRA rank (default: 32)
        alpha: LoRA alpha scaling factor (default: 32.0)
        dropout: LoRA dropout rate (default: 0.1)
        target_modules: Which attention modules to apply LoRA to
    """

    name = "standard"

    def __init__(
        self,
        rank: int = 32,
        alpha: float = 32.0,
        dropout: float = 0.1,
        target_modules: List[str] = None,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules or ["to_q", "to_k", "to_v", "to_out"]

    def apply(self, model: nn.Module, **kwargs) -> None:
        """
        Apply Standard LoRA to the model.

        Uses existing apply_lora_to_unet() from lora_unet.py.

        Args:
            model: UNet model to apply LoRA to
        """
        print(f"[StandardLoRA] Applying LoRA (rank={self.rank}, alpha={self.alpha}, dropout={self.dropout})")
        apply_lora_to_unet(
            unet=model,
            rank=self.rank,
            alpha=self.alpha,
            dropout=self.dropout,
        )
        stats = self.count_parameters(model)
        print(f"[StandardLoRA] Trainable params: {stats['trainable']:,} ({stats['trainable_percentage']:.2f}%)")

    def get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """
        Return trainable LoRA parameters.

        Uses existing get_lora_parameters() from lora_unet.py.

        Args:
            model: Model containing LoRA layers

        Returns:
            List of trainable LoRA parameters
        """
        return get_lora_parameters(model)

    def get_config(self) -> Dict[str, Any]:
        """Return LoRA configuration."""
        return {
            "name": self.name,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


# Register with the LoRA registry
def _register():
    """Register StandardLoRA with the global registry."""
    from . import register_lora
    register_lora("standard")(StandardLoRA)


# Auto-register when module is imported
_register()
