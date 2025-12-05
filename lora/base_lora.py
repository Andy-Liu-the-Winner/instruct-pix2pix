"""
Base LoRA class - Abstract interface for all LoRA implementations.

All LoRA types (standard, intrinsic, drag, etc.) should inherit from this class.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import List, Dict, Any


class BaseLoRA(ABC):
    """
    Abstract base class for all LoRA implementations.

    Subclasses must implement:
        - apply(): Inject LoRA into model
        - get_trainable_params(): Return trainable parameters
        - save(): Save LoRA weights
        - load(): Load LoRA weights
    """

    name: str = "base"

    @abstractmethod
    def apply(self, model: nn.Module, **kwargs) -> None:
        """
        Apply LoRA to the model.

        Args:
            model: The model to apply LoRA to (typically UNet)
            **kwargs: Additional arguments specific to LoRA type
        """
        pass

    @abstractmethod
    def get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """
        Return list of trainable LoRA parameters.

        Args:
            model: The model containing LoRA layers

        Returns:
            List of trainable parameters for optimizer
        """
        pass

    def save(self, model: nn.Module, path: str) -> None:
        """
        Save LoRA weights to file.

        Args:
            model: The model containing LoRA layers
            path: Path to save weights
        """
        lora_state_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad and ('lora_' in name or 'intrinsic_' in name):
                lora_state_dict[name] = param.data.clone()

        torch.save({
            'lora_type': self.name,
            'config': self.get_config(),
            'state_dict': lora_state_dict,
        }, path)
        print(f"[{self.name}] Saved {len(lora_state_dict)} LoRA tensors to {path}")

    def load(self, model: nn.Module, path: str) -> None:
        """
        Load LoRA weights from file.

        Args:
            model: The model to load LoRA weights into
            path: Path to load weights from
        """
        checkpoint = torch.load(path, map_location='cpu')

        if checkpoint.get('lora_type') != self.name:
            print(f"Warning: Loading {checkpoint.get('lora_type')} weights into {self.name}")

        state_dict = checkpoint['state_dict']
        model_state = model.state_dict()

        loaded = 0
        for name, param in state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)
                loaded += 1

        print(f"[{self.name}] Loaded {loaded}/{len(state_dict)} LoRA tensors from {path}")

    def get_config(self) -> Dict[str, Any]:
        """
        Return LoRA configuration as dictionary.

        Returns:
            Configuration dict
        """
        return {"name": self.name}

    def count_parameters(self, model: nn.Module) -> Dict[str, Any]:
        """
        Count trainable and frozen parameters.

        Args:
            model: The model to analyze

        Returns:
            Dictionary with parameter counts
        """
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        total = trainable + frozen

        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_percentage': 100 * trainable / total if total > 0 else 0
        }

    def __repr__(self) -> str:
        config = self.get_config()
        config_str = ", ".join(f"{k}={v}" for k, v in config.items() if k != "name")
        return f"{self.__class__.__name__}({config_str})"
