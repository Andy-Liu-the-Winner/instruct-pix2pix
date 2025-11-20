import sys
sys.path.append("./stable_diffusion")

import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from lora_unet import LoRALinear

def load_model(checkpoint_path):
    config = OmegaConf.load("configs/train.yaml")

    # Don't load base checkpoint, just need structure
    if 'ckpt_path' in config.model.params:
        config.model.params.ckpt_path = None

    model = instantiate_from_config(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model, checkpoint

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'pct': 100 * trainable / total if total > 0 else 0
    }

def get_lora_layers(model):
    layers = []
    total_params = 0

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear) and module.is_lora():
            params = module.lora_A.numel() + module.lora_B.numel()
            layers.append({
                'name': name,
                'rank': module.lora_rank,
                'in': module.in_features,
                'out': module.out_features,
                'params': params
            })
            total_params += params

    return layers, total_params

if __name__ == "__main__":
    checkpoint_paths = [
        "logs/train_hq_edit_lora_experiment_optimizer_fixed_trainable_for_lora/checkpoints/last.ckpt",
        "logs/train_hq_edit_lora_experiment_optimizer_fixed_trainable_for_lora/checkpoints/epoch=000019.ckpt",
    ]

    checkpoint_path = None
    for path in checkpoint_paths:
        try:
            torch.load(path, map_location="cpu")
            checkpoint_path = path
            break
        except:
            continue

    if not checkpoint_path:
        print("Error: Could not load checkpoint")
        exit(1)

    print("Loading model...")
    model, checkpoint = load_model(checkpoint_path)

    stats = count_parameters(model)
    lora_layers, lora_params = get_lora_layers(model)

    print(f"\nParameter Stats:")
    print(f"  Total: {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,} ({stats['pct']:.2f}%)")
    print(f"  Frozen: {stats['frozen']:,}")

    print(f"\nLoRA Info:")
    print(f"  Layers: {len(lora_layers)}")
    print(f"  Parameters: {lora_params:,}")
    if lora_layers:
        print(f"  Rank: {lora_layers[0]['rank']}")

    # Group by layer type
    layer_types = {}
    for layer in lora_layers:
        layer_type = layer['name'].split('.')[-1]
        if layer_type not in layer_types:
            layer_types[layer_type] = []
        layer_types[layer_type].append(layer)

    print(f"\nLayer Breakdown:")
    for ltype in sorted(layer_types.keys()):
        layers = layer_types[ltype]
        total = sum(l['params'] for l in layers)
        print(f"  {ltype}: {len(layers)} layers, {total:,} params")
        for layer in layers[:2]:  # Show first 2
            print(f"    {layer['name'][-40:]}: {layer['params']:,}")
        if len(layers) > 2:
            print(f"    ... {len(layers)-2} more")
