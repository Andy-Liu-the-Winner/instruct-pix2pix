"""
LoRA Combiner - Combine multiple LoRA types on a single model.

Allows running Standard LoRA, Intrinsic LoRA, and other LoRA types in parallel.

Example:
    combiner = LoRACombiner([
        {"type": "standard", "rank": 32, "alpha": 32.0},
        {"type": "intrinsic", "rank": 4, "intrinsic_types": ["depth", "normals"]},
    ])
    combiner.apply_all(model)
    params = combiner.get_all_trainable_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-4)
"""

import torch.nn as nn
from typing import List, Dict, Any

from .base_lora import BaseLoRA


class LoRACombiner:
    """
    Combine multiple LoRAs on a single model.

    Each LoRA type can operate on different parts of the model or use
    different adaptation strategies, allowing them to work in parallel.

    Args:
        lora_configs: List of config dicts, each with 'type' key and LoRA-specific params

    Example configs:
        [
            {"type": "standard", "rank": 32, "alpha": 32.0, "dropout": 0.1},
            {"type": "intrinsic", "rank": 4, "intrinsic_types": ["depth"]},
        ]
    """

    def __init__(self, lora_configs: List[Dict[str, Any]]):
        # Import here to avoid circular imports
        from . import get_lora

        self.loras: List[BaseLoRA] = []
        self.configs = lora_configs

        for config in lora_configs:
            config = config.copy()  # Don't modify original
            lora_type = config.pop("type")

            # Get LoRA class and instantiate
            lora_cls = get_lora(lora_type)
            lora_instance = lora_cls(**config)
            self.loras.append(lora_instance)

        print(f"[LoRACombiner] Created combiner with {len(self.loras)} LoRA types:")
        for lora in self.loras:
            print(f"  - {lora}")

    def apply_all(self, model: nn.Module, **kwargs) -> None:
        """
        Apply all LoRAs to the model.

        Args:
            model: Model to apply LoRAs to (typically UNet)
            **kwargs: Additional arguments passed to each LoRA
        """
        print(f"\n[LoRACombiner] Applying {len(self.loras)} LoRA types...")

        for lora in self.loras:
            lora.apply(model, **kwargs)

        print(f"[LoRACombiner] All LoRAs applied successfully\n")

    def get_all_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """
        Get trainable parameters from all LoRAs.

        Args:
            model: Model containing LoRA layers

        Returns:
            Combined list of trainable parameters
        """
        all_params = []
        seen_ids = set()  # Avoid duplicates

        for lora in self.loras:
            params = lora.get_trainable_params(model)
            for p in params:
                if id(p) not in seen_ids:
                    all_params.append(p)
                    seen_ids.add(id(p))

        return all_params

    def save_all(self, model: nn.Module, path_prefix: str) -> None:
        """
        Save all LoRA weights.

        Each LoRA type is saved to a separate file.

        Args:
            model: Model containing LoRA layers
            path_prefix: Prefix for save paths (e.g., "checkpoints/lora")
        """
        for lora in self.loras:
            path = f"{path_prefix}_{lora.name}.pt"
            lora.save(model, path)

    def load_all(self, model: nn.Module, path_prefix: str) -> None:
        """
        Load all LoRA weights.

        Args:
            model: Model to load LoRA weights into
            path_prefix: Prefix for load paths
        """
        for lora in self.loras:
            path = f"{path_prefix}_{lora.name}.pt"
            lora.load(model, path)

    def get_combined_config(self) -> Dict[str, Any]:
        """Return combined configuration of all LoRAs."""
        return {
            "num_loras": len(self.loras),
            "loras": [lora.get_config() for lora in self.loras],
        }

    def count_all_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """Count parameters for all LoRAs combined."""
        params = self.get_all_trainable_params(model)
        trainable = sum(p.numel() for p in params)
        total = sum(p.numel() for p in model.parameters())

        return {
            "trainable": trainable,
            "total": total,
            "trainable_percentage": 100 * trainable / total if total > 0 else 0,
            "per_lora": {
                lora.name: sum(p.numel() for p in lora.get_trainable_params(model))
                for lora in self.loras
            }
        }

    def __repr__(self) -> str:
        lora_names = [lora.name for lora in self.loras]
        return f"LoRACombiner(loras={lora_names})"

    def __len__(self) -> int:
        return len(self.loras)

    def __iter__(self):
        return iter(self.loras)
