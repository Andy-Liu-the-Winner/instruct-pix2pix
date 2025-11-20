import torch
import torch.nn as nn
import math
from typing import List


class LoRALinear(nn.Linear):
    """
    LoRA-enhanced Linear layer that extends nn.Linear.

    Applies low-rank adaptation: y = Wx + (lora_scaling) * B @ A @ x
    where W is the original weight, and A, B are trainable low-rank matrices.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True,
                 device=None,
                 dtype=None,
                 lora_rank: int = 0,
                 lora_alpha: float = 0.0,
                 lora_dropout: float = 0.0,
                ) -> None:

        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.has_weights_merged = False

        if lora_rank > 0:
            self.lora_rank = lora_rank
            self.lora_alpha = lora_alpha
            self.lora_dropout = nn.Dropout(p=lora_dropout)
            self.lora_scaling = lora_alpha / lora_rank

            factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}
            self.lora_A = nn.Parameter(torch.empty((lora_rank, in_features), **factory_kwargs))
            self.lora_B = nn.Parameter(torch.empty((out_features, lora_rank), **factory_kwargs))

            self.lora_A.requires_grad = False
            self.lora_B.requires_grad = False

            self.reset_parameters()

    def is_lora(self) -> bool:
        return hasattr(self, 'lora_A')

    def reset_parameters(self) -> None:
        nn.Linear.reset_parameters(self)
        if self.is_lora():
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = super().forward(input)

        if self.is_lora() and not self.has_weights_merged:
            dropped = self.lora_dropout(input)
            lora_update = dropped @ self.lora_A.t()
            lora_update = lora_update @ self.lora_B.t()
            result = result + self.lora_scaling * lora_update

        return result

    def train(self, mode: bool = True) -> "LoRALinear":
        super().train(mode)

        if self.is_lora():
            if mode and self.has_weights_merged:
                lora_weight = torch.matmul(self.lora_B, self.lora_A)
                self.weight.data -= self.lora_scaling * lora_weight
                self.has_weights_merged = False
            elif not mode and not self.has_weights_merged:
                lora_weight = torch.matmul(self.lora_B, self.lora_A)
                self.weight.data += self.lora_scaling * lora_weight
                self.has_weights_merged = True

        return self

    def eval(self) -> "LoRALinear":
        return self.train(False)

    def extra_repr(self) -> str:
        out = nn.Linear.extra_repr(self)
        if self.is_lora():
            out += f', lora_rank={self.lora_A.shape[0]}, lora_scaling={self.lora_scaling}, lora_dropout={self.lora_dropout.p}'
        return out

# added helper functions for LoRA application
def _replace_linear_with_lora(module, name, rank, alpha, dropout):
    """
    Helper function to replace a Linear layer with LoRALinear.
    """
    original_linear = getattr(module, name)
    if not isinstance(original_linear, nn.Linear):
        return

    lora_linear = LoRALinear(
        in_features=original_linear.in_features,
        out_features=original_linear.out_features,
        bias=original_linear.bias is not None,
        device=original_linear.weight.device,
        dtype=original_linear.weight.dtype,
        lora_rank=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
    )

    with torch.no_grad():
        lora_linear.weight.copy_(original_linear.weight)
        if original_linear.bias is not None:
            lora_linear.bias.copy_(original_linear.bias)

    setattr(module, name, lora_linear)


def apply_lora_to_unet(unet: nn.Module, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0) -> None:
    """
    Inject LoRA layers into all Attention modules in the UNet and
    disable gradient checkpointing for stable LoRA training.
    """

    # --- 0) Disable gradient checkpointing everywhere in the UNet ---
    disabled = 0
    for module in unet.modules():
        if hasattr(module, "checkpoint"):
            module.checkpoint = False
            disabled += 1
        if hasattr(module, "use_checkpoint"):
            module.use_checkpoint = False
            disabled += 1
    print(f"[LoRA] Disabled gradient checkpointing on {disabled} attributes.")

    # --- 1) Patch attention projections with LoRA ---
    for module in unet.modules():
        if module.__class__.__name__ in ["Attention", "CrossAttention"]:
            if hasattr(module, "to_q"):
                _replace_linear_with_lora(module, "to_q", rank, alpha, dropout)
            if hasattr(module, "to_k"):
                _replace_linear_with_lora(module, "to_k", rank, alpha, dropout)
            if hasattr(module, "to_v"):
                _replace_linear_with_lora(module, "to_v", rank, alpha, dropout)

            if hasattr(module, "to_out"):
                if isinstance(module.to_out, (nn.ModuleList, nn.Sequential)):
                    if len(module.to_out) > 0 and isinstance(module.to_out[0], nn.Linear):
                        _replace_linear_with_lora(module.to_out, "0", rank, alpha, dropout)

    # --- 2) Freeze base weights, unfreeze only LoRA ---
    mark_only_lora_as_trainable(unet)

def mark_only_lora_as_trainable(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters except LoRA parameters.
    Works because gradient checkpointing is disabled (use_checkpoint=False in config).
    """
    # Freeze ALL parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze ONLY LoRA parameters
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.is_lora():
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    return model



def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    Collect all LoRA parameters from the model.

    Args:
        model: The model containing LoRA layers

    Returns:
        List of trainable LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear) and module.is_lora():
            lora_params.append(module.lora_A)
            lora_params.append(module.lora_B)
    return lora_params


def count_parameters(model: nn.Module) -> dict:
    """
    Count trainable and frozen parameters in the model.

    Args:
        model: The model to analyze

    Returns:
        Dictionary with trainable and frozen parameter counts
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


if __name__ == '__main__':
    from diffusers import StableDiffusionInstructPix2PixPipeline

    print("Loading InstructPix2Pix model...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float32,
        safety_checker=None
    )
    unet = pipe.unet

    print("\nBefore LoRA injection:")
    stats_before = count_parameters(unet)
    print(f"  Trainable parameters: {stats_before['trainable']:,}")
    print(f"  Frozen parameters: {stats_before['frozen']:,}")
    print(f"  Total parameters: {stats_before['total']:,}")

    print("\nApplying LoRA to UNet (rank=4, alpha=1.0, dropout=0.0)...")
    apply_lora_to_unet(unet, rank=4, alpha=1.0, dropout=0.0)

    print("\nAfter LoRA injection:")
    stats_after = count_parameters(unet)
    print(f"  Trainable parameters: {stats_after['trainable']:,}")
    print(f"  Frozen parameters: {stats_after['frozen']:,}")
    print(f"  Total parameters: {stats_after['total']:,}")
    print(f"  Trainable percentage: {stats_after['trainable_percentage']:.4f}%")

    lora_params = get_lora_parameters(unet)
    print(f"\nNumber of LoRA parameter tensors: {len(lora_params)}")
    print(f"Total LoRA parameters: {sum(p.numel() for p in lora_params):,}")

    print("\nLoRA injection complete. UNet is ready for fine-tuning.")
    print("\nExample usage in training:")
    print("  optimizer = torch.optim.AdamW(get_lora_parameters(unet), lr=1e-4)")
    print("  unet.train()  # Weights will be unmerged for training")
    print("  unet.eval()   # Weights will be merged for efficient inference")
