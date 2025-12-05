"""
Intrinsic LoRA - Extract scene intrinsics (depth, normals, albedo, shading).

Based on: "Generative Models: What Do They Know? Do They Know Things? Let's Find Out!"
Paper: https://arxiv.org/abs/2311.17137
Reference: https://github.com/duxiaodan/intrinsic-lora

Key differences from Standard LoRA:
- Uses attention processor replacement (not linear layer replacement)
- Supports auxiliary intrinsic prediction heads
- Can predict multiple intrinsic types simultaneously
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import math

from .base_lora import BaseLoRA


class LoRALayer(nn.Module):
    """
    Single LoRA layer: input -> down_proj -> up_proj -> output

    Implements: delta_W = B @ A where A is (rank, in_features), B is (out_features, rank)
    """

    def __init__(self, in_features: int, out_features: int, rank: int = 4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # Down projection: in_features -> rank
        self.down = nn.Linear(in_features, rank, bias=False)
        # Up projection: rank -> out_features
        self.up = nn.Linear(rank, out_features, bias=False)

        # Initialize: A with kaiming, B with zeros (so LoRA starts as identity)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class LoRAAttnProcessor(nn.Module):
    """
    Attention processor with LoRA weights.

    Replaces standard attention computation with LoRA-augmented version.
    LoRA is applied to Q, K, V, and output projections.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: Optional[int] = None,
        rank: int = 4,
        alpha: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim or hidden_size
        self.rank = rank
        self.scale = alpha / rank

        # LoRA layers for Q, K, V, Out
        self.to_q_lora = LoRALayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALayer(self.cross_attention_dim, hidden_size, rank)
        self.to_v_lora = LoRALayer(self.cross_attention_dim, hidden_size, rank)
        self.to_out_lora = LoRALayer(hidden_size, hidden_size, rank)

    def forward(
        self,
        attn,  # Original attention module
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with LoRA augmentation.

        Args:
            attn: Original attention module (has to_q, to_k, to_v, to_out)
            hidden_states: Input tensor
            encoder_hidden_states: Cross-attention context (None for self-attention)
            attention_mask: Optional attention mask
        """
        batch_size, sequence_length, _ = hidden_states.shape

        # Query with LoRA
        query = attn.to_q(hidden_states) + self.scale * self.to_q_lora(hidden_states)

        # Key and Value (from encoder states for cross-attention)
        key_input = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = attn.to_k(key_input) + self.scale * self.to_k_lora(key_input)
        value = attn.to_v(key_input) + self.scale * self.to_v_lora(key_input)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Attention scores
        scale = 1 / math.sqrt(head_dim)
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attention output
        hidden_states = torch.matmul(attn_weights, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, inner_dim)

        # Output projection with LoRA
        hidden_states = attn.to_out[0](hidden_states) + self.scale * self.to_out_lora(hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        return hidden_states


class IntrinsicHead(nn.Module):
    """
    Lightweight head for intrinsic property prediction.

    Takes UNet features and predicts intrinsic maps (depth, normals, etc.)
    """

    def __init__(self, in_channels: int = 320, out_channels: int = 4, hidden_channels: int = 160):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class IntrinsicLoRA(BaseLoRA):
    """
    Intrinsic LoRA for scene property extraction.

    Extracts intrinsic scene properties (depth, normals, albedo, shading) from
    diffusion model features using lightweight LoRA adaptation.

    Key features:
    - Attention processor replacement (LoRAAttnProcessor)
    - Optional auxiliary heads for explicit intrinsic prediction
    - MSE loss for intrinsic targets

    Args:
        rank: LoRA rank (default: 4, very lightweight)
        alpha: LoRA alpha scaling (default: 4.0)
        intrinsic_types: List of intrinsics to extract ["depth", "normals", "albedo", "shading"]
        use_auxiliary_heads: Whether to add prediction heads
        auxiliary_loss_weight: Weight for auxiliary intrinsic loss
    """

    name = "intrinsic"
    SUPPORTED_INTRINSICS = ["depth", "normals", "albedo", "shading"]

    def __init__(
        self,
        rank: int = 4,
        alpha: float = 4.0,
        intrinsic_types: List[str] = None,
        use_auxiliary_heads: bool = False,
        auxiliary_loss_weight: float = 0.1,
    ):
        self.rank = rank
        self.alpha = alpha
        self.intrinsic_types = intrinsic_types or ["depth", "normals"]
        self.use_auxiliary_heads = use_auxiliary_heads
        self.auxiliary_loss_weight = auxiliary_loss_weight

        # Validate intrinsic types
        for itype in self.intrinsic_types:
            if itype not in self.SUPPORTED_INTRINSICS:
                raise ValueError(f"Unknown intrinsic type: {itype}. Supported: {self.SUPPORTED_INTRINSICS}")

        # Storage for attention processors
        self.attn_processors: Dict[str, LoRAAttnProcessor] = {}
        # Storage for intrinsic heads
        self.intrinsic_heads: Optional[nn.ModuleDict] = None

    def apply(self, model: nn.Module, **kwargs) -> None:
        """
        Apply Intrinsic LoRA to the model.

        Replaces attention modules with LoRA-augmented processors.

        Args:
            model: UNet model to apply LoRA to
        """
        print(f"[IntrinsicLoRA] Applying LoRA (rank={self.rank}, alpha={self.alpha})")
        print(f"[IntrinsicLoRA] Intrinsic types: {self.intrinsic_types}")

        # Disable gradient checkpointing
        self._disable_checkpointing(model)

        # Replace attention processors
        self._replace_attention_processors(model)

        # Optionally add intrinsic heads
        if self.use_auxiliary_heads:
            self._add_intrinsic_heads(model)

        # Freeze non-LoRA parameters
        self._freeze_non_lora(model)

        stats = self.count_parameters(model)
        print(f"[IntrinsicLoRA] Trainable params: {stats['trainable']:,} ({stats['trainable_percentage']:.4f}%)")

    def _disable_checkpointing(self, model: nn.Module) -> None:
        """Disable gradient checkpointing for stable LoRA training."""
        disabled = 0
        for module in model.modules():
            if hasattr(module, "checkpoint"):
                module.checkpoint = False
                disabled += 1
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = False
                disabled += 1
        if disabled > 0:
            print(f"[IntrinsicLoRA] Disabled checkpointing on {disabled} attributes")

    def _replace_attention_processors(self, model: nn.Module) -> None:
        """Replace attention modules with LoRA-augmented processors."""
        attn_count = 0

        for name, module in model.named_modules():
            if module.__class__.__name__ in ["Attention", "CrossAttention"]:
                # Get dimensions from existing attention module
                hidden_size = module.to_q.in_features
                cross_attention_dim = None

                # Check if this is cross-attention (different key/value dim)
                if hasattr(module, 'to_k') and module.to_k.in_features != hidden_size:
                    cross_attention_dim = module.to_k.in_features

                # Create LoRA processor
                processor = LoRAAttnProcessor(
                    hidden_size=hidden_size,
                    cross_attention_dim=cross_attention_dim,
                    rank=self.rank,
                    alpha=self.alpha,
                )

                # Store processor (will be used during forward)
                self.attn_processors[name] = processor

                # Attach processor to module for easy access
                module.lora_processor = processor

                attn_count += 1

        print(f"[IntrinsicLoRA] Replaced {attn_count} attention modules with LoRA processors")

    def _add_intrinsic_heads(self, model: nn.Module) -> None:
        """Add lightweight heads for intrinsic prediction."""
        self.intrinsic_heads = nn.ModuleDict({
            itype: IntrinsicHead(in_channels=320, out_channels=4)
            for itype in self.intrinsic_types
        })

        # Attach to model for easy access
        model.intrinsic_heads = self.intrinsic_heads
        print(f"[IntrinsicLoRA] Added {len(self.intrinsic_types)} intrinsic heads")

    def _freeze_non_lora(self, model: nn.Module) -> None:
        """Freeze all parameters except LoRA parameters."""
        # Freeze all
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA processors
        for processor in self.attn_processors.values():
            for param in processor.parameters():
                param.requires_grad = True

        # Unfreeze intrinsic heads
        if self.intrinsic_heads is not None:
            for head in self.intrinsic_heads.values():
                for param in head.parameters():
                    param.requires_grad = True

    def get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """Return trainable LoRA parameters."""
        params = []

        # LoRA processor parameters
        for processor in self.attn_processors.values():
            params.extend(processor.parameters())

        # Intrinsic head parameters
        if self.intrinsic_heads is not None:
            for head in self.intrinsic_heads.values():
                params.extend(head.parameters())

        return params

    def compute_auxiliary_loss(
        self,
        features: torch.Tensor,
        targets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute auxiliary loss for intrinsic prediction.

        Args:
            features: UNet intermediate features
            targets: Dict of intrinsic targets {"depth": tensor, "normals": tensor, ...}

        Returns:
            Auxiliary loss tensor
        """
        if not self.use_auxiliary_heads or self.intrinsic_heads is None:
            return torch.tensor(0.0)

        loss = 0.0
        for itype in self.intrinsic_types:
            if itype in targets and targets[itype] is not None:
                pred = self.intrinsic_heads[itype](features)
                target = targets[itype]
                loss += F.mse_loss(pred, target)

        return self.auxiliary_loss_weight * loss

    def get_config(self) -> Dict[str, Any]:
        """Return LoRA configuration."""
        return {
            "name": self.name,
            "rank": self.rank,
            "alpha": self.alpha,
            "intrinsic_types": self.intrinsic_types,
            "use_auxiliary_heads": self.use_auxiliary_heads,
            "auxiliary_loss_weight": self.auxiliary_loss_weight,
        }


# Register with the LoRA registry
def _register():
    """Register IntrinsicLoRA with the global registry."""
    from . import register_lora
    register_lora("intrinsic")(IntrinsicLoRA)


# Auto-register when module is imported
_register()
