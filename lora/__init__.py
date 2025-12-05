"""
Modular LoRA Framework for InstructPix2Pix
==========================================

A flexible framework supporting multiple LoRA types that can run in parallel:

- **standard**: Standard LoRA on attention Q, K, V, Out projections
  (wraps existing lora_unet.py implementation)
- **intrinsic**: Intrinsic LoRA for scene property extraction (depth, normals, etc.)
- (future) drag: Drag-based manipulation LoRA
- (future) esp: Edge/structure preservation LoRA

Quick Start:
------------
    # Single LoRA
    from lora import get_lora
    lora = get_lora("standard")(rank=32, alpha=32.0)
    lora.apply(model)
    params = lora.get_trainable_params(model)

    # Multiple LoRAs in parallel
    from lora import LoRACombiner
    combiner = LoRACombiner([
        {"type": "standard", "rank": 32, "alpha": 32.0},
        {"type": "intrinsic", "rank": 4, "intrinsic_types": ["depth", "normals"]},
    ])
    combiner.apply_all(model)
    params = combiner.get_all_trainable_params(model)

Available LoRA Types:
---------------------
    >>> from lora import list_loras
    >>> list_loras()
    ['standard', 'intrinsic']
"""

from typing import Dict, Type, List

# Global registry for LoRA implementations
_LORA_REGISTRY: Dict[str, Type] = {}


def register_lora(name: str):
    """
    Decorator to register a LoRA class.

    Usage:
        @register_lora("my_lora")
        class MyLoRA(BaseLoRA):
            ...

    Args:
        name: Unique name for this LoRA type
    """
    def decorator(cls):
        if name in _LORA_REGISTRY:
            print(f"[LoRA Registry] Warning: Overwriting existing LoRA '{name}'")
        _LORA_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


def get_lora(name: str):
    """
    Get LoRA class by name.

    Args:
        name: Name of the registered LoRA

    Returns:
        LoRA class

    Raises:
        ValueError: If LoRA name is not registered

    Example:
        >>> StandardLoRA = get_lora("standard")
        >>> lora = StandardLoRA(rank=32)
    """
    if name not in _LORA_REGISTRY:
        available = list(_LORA_REGISTRY.keys())
        raise ValueError(f"Unknown LoRA: '{name}'. Available: {available}")
    return _LORA_REGISTRY[name]


def list_loras() -> List[str]:
    """
    List all registered LoRA names.

    Returns:
        List of registered LoRA type names

    Example:
        >>> list_loras()
        ['standard', 'intrinsic']
    """
    return list(_LORA_REGISTRY.keys())


# Import base class first
from .base_lora import BaseLoRA

# Import LoRA implementations (they auto-register via decorator)
from .intrinsic_lora import IntrinsicLoRA, LoRALayer, LoRAAttnProcessor, IntrinsicHead

# Import combiner
from .lora_combiner import LoRACombiner

# Import TORE (Transforming Original Relations Effectively)
from .tore import apply_tore, apply_tore_batch, TOREPreprocessor

# Re-export from original lora_unet.py for backward compatibility
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lora_unet import (
    LoRALinear,
    apply_lora_to_unet,
    mark_only_lora_as_trainable,
    get_lora_parameters,
    count_parameters,
)

__all__ = [
    # Registry functions
    "register_lora",
    "get_lora",
    "list_loras",
    # Base class
    "BaseLoRA",
    # LoRA implementations
    "IntrinsicLoRA",
    # Intrinsic LoRA components
    "LoRALayer",
    "LoRAAttnProcessor",
    "IntrinsicHead",
    # Combiner
    "LoRACombiner",
    # TORE (ESPLoRA)
    "apply_tore",
    "apply_tore_batch",
    "TOREPreprocessor",
    # Legacy exports (from lora_unet.py)
    "LoRALinear",
    "apply_lora_to_unet",
    "mark_only_lora_as_trainable",
    "get_lora_parameters",
    "count_parameters",
]


def _print_registry_info():
    """Print info about registered LoRAs (for debugging)."""
    print(f"[LoRA Registry] {len(_LORA_REGISTRY)} LoRA types registered:")
    for name, cls in _LORA_REGISTRY.items():
        print(f"  - {name}: {cls.__name__}")


# Uncomment for debugging:
# _print_registry_info()
