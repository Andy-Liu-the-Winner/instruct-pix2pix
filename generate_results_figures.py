import sys
sys.path.append("./stable_diffusion")

import torch
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np
import os
from pathlib import Path

MODELS = [
    ("checkpoints/instruct-pix2pix-00-22000.ckpt", "configs/train.yaml", "Baseline"),
    ("logs/train_spatial_lora_test/checkpoints/last.ckpt", "configs/train.yaml", "LoRA_r32"),
    ("logs/train_esplora_train_esplora/checkpoints/last.ckpt", "configs/train_esplora.yaml", "ESPLoRA_r64"),
]

# Output directory
OUTPUT_DIR = "paper_results"

# Test images - select diverse, high-quality examples
TEST_IMAGES = [
    ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "cat"),
    ("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512", "man"),
    ("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512", "woman"),
    ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512", "mountain"),
]

# Prompts grouped by category
PROMPT_GROUPS = {
    "spatial": [
        "Move the subject to the left side",
        "Move the subject to the right side",
        "Place a ball to the left of the subject",
        "Add a bird above the subject",
    ],
    "appearance": [
        "Add sunglasses",
        "Put a hat on the subject",
        "Make it look like a painting",
        "Change to winter with snow",
    ],
    "combined": [
        "Move left and add sunglasses",
        "Make it a watercolor and add flowers",
    ]
}

STEPS = 100
GENERATE_GRIDS = True  # Create comparison grids


def load_image_from_url(url):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_model(checkpoint_path, config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.cuda()
    model.eval()
    return model

@torch.no_grad()
def edit_image(model, input_image, prompt, steps=100):
    import k_diffusion as K
    from einops import rearrange, repeat
    from torch import autocast

    with autocast("cuda"):
        width, height = input_image.size
        resolution = 512

        if max(width, height) > resolution:
            if width > height:
                new_width = resolution
                new_height = int(height * resolution / width)
            else:
                new_height = resolution
                new_width = int(width * resolution / height)
            input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        else:
            new_width, new_height = width, height

        pad_width = ((new_width + 63) // 64) * 64
        pad_height = ((new_height + 63) // 64) * 64

        if pad_width != new_width or pad_height != new_height:
            padded = Image.new("RGB", (pad_width, pad_height), (0, 0, 0))
            padded.paste(input_image, ((pad_width - new_width) // 2, (pad_height - new_height) // 2))
            input_image = padded

        image_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        image_tensor = rearrange(image_tensor, "h w c -> 1 c h w").cuda()
        image_latent = model.encode_first_stage(image_tensor).mode()

        cond = {
            "c_crossattn": [model.get_learned_conditioning([prompt])],
            "c_concat": [image_latent]
        }
        null_token = model.get_learned_conditioning([""])
        uncond = {
            "c_crossattn": [null_token],
            "c_concat": [torch.zeros_like(image_latent)]
        }

        model_wrap = K.external.CompVisDenoiser(model)

        class CFGDenoiser:
            def __init__(self, inner_model):
                self.inner_model = inner_model

            def __call__(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
                cfg_z = repeat(z, "1 ... -> n ...", n=3)
                cfg_sigma = repeat(sigma, "1 ... -> n ...", n=3)
                cond_concat = torch.cat(cond["c_concat"], dim=1)
                uncond_concat = torch.cat(uncond["c_concat"], dim=1)
                cfg_cond = {
                    "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
                    "c_concat": [torch.cat([cond_concat, cond_concat, uncond_concat])],
                }
                out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
                return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)

        model_wrap_cfg = CFGDenoiser(model_wrap)
        sigmas = model_wrap.get_sigmas(steps)
        z = torch.randn_like(cond["c_concat"][0]) * sigmas[0]
        z = K.sampling.sample_euler_ancestral(model_wrap_cfg, z, sigmas,
                                              extra_args={"cond": cond, "uncond": uncond,
                                                         "text_cfg_scale": 7.5, "image_cfg_scale": 1.5})

        x = model.decode_first_stage(z)
        x = torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)
        x = 255.0 * rearrange(x, "1 c h w -> h w c")
        return Image.fromarray(x.type(torch.uint8).cpu().numpy())

def create_comparison_grid(image_name, prompt, model_results, input_image):
    """
    Create grid: [Input | Baseline | LoRA | ESPLoRA]
    """
    cell_size = 256
    header_height = 80
    n_models = len(model_results)

    grid_width = (n_models + 1) * cell_size  # +1 for input
    grid_height = header_height + cell_size

    grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
        font_title = font

    # Title (prompt)
    prompt_text = prompt[:60] + "..." if len(prompt) > 60 else prompt
    draw.text((10, 10), f"Prompt: {prompt_text}", fill=(0, 0, 0), font=font_title)
    draw.text((10, 35), f"Image: {image_name}", fill=(100, 100, 100), font=font)

    # Column headers
    draw.text((cell_size//2 - 20, header_height - 25), "Input", fill=(0, 0, 0), font=font)
    for col, (model_name, _) in enumerate(model_results):
        x = (col + 1) * cell_size + cell_size//4
        draw.text((x, header_height - 25), model_name, fill=(0, 0, 0), font=font)

    # Paste images
    input_resized = input_image.resize((cell_size, cell_size), Image.LANCZOS)
    grid.paste(input_resized, (0, header_height))

    for col, (model_name, output_img) in enumerate(model_results):
        output_resized = output_img.resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(output_resized, ((col + 1) * cell_size, header_height))

    return grid

# ============== MAIN ==============

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create subdirectories for organization
    for group in PROMPT_GROUPS.keys():
        os.makedirs(os.path.join(OUTPUT_DIR, group), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "grids"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "individual"), exist_ok=True)

    # Load test images
    print("Loading test images...")
    test_images = []
    for url, name in TEST_IMAGES:
        try:
            img = load_image_from_url(url)
            test_images.append((img, name))
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Count total generations
    total = len(test_images) * sum(len(prompts) for prompts in PROMPT_GROUPS.values()) * len(MODELS)
    print(f"\nTotal generations: {total} ({len(test_images)} images × {sum(len(p) for p in PROMPT_GROUPS.values())} prompts × {len(MODELS)} models)")
    print(f"Estimated time: {total * 5 / 60:.1f} minutes\n")

    # Generate results grouped by category
    for group_name, prompts in PROMPT_GROUPS.items():
        print(f"\n{'='*70}")
        print(f"GROUP: {group_name.upper()}")
        print(f"{'='*70}\n")

        for img, img_name in test_images:
            for prompt_idx, prompt in enumerate(prompts):
                print(f"[{img_name}] {prompt[:50]}...")

                model_results = []

                # Generate with each model
                for ckpt_path, config_path, model_name in MODELS:
                    if not os.path.exists(ckpt_path):
                        print(f"  ✗ {model_name}: checkpoint not found")
                        continue

                    try:
                        # Load model
                        model = load_model(ckpt_path, config_path)

                        # Generate
                        output = edit_image(model, img.copy(), prompt, steps=STEPS)

                        # Save individual
                        safe_prompt = prompt[:30].replace(" ", "_")
                        output_path = os.path.join(OUTPUT_DIR, "individual",
                                                   f"{group_name}_{img_name}_{model_name}_{safe_prompt}.png")
                        output.save(output_path)

                        model_results.append((model_name, output))
                        print(f"  ✓ {model_name}")

                        # Free memory
                        del model
                        torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"  ✗ {model_name}: {e}")

                # Create comparison grid
                if GENERATE_GRIDS and model_results:
                    try:
                        grid = create_comparison_grid(img_name, prompt, model_results, img)
                        grid_path = os.path.join(OUTPUT_DIR, "grids",
                                                f"{group_name}_{img_name}_p{prompt_idx:02d}.png")
                        grid.save(grid_path)
                        print(f"    yep Grid saved: {grid_path}")
                    except Exception as e:
                        print(f"   nope, Grid error: {e}")

    print(f"DONE!")
    print(f"{'='*70}")
    print(f"Results saved to:")
    print(f"  - {OUTPUT_DIR}/grids/        (comparison grids)")
    print(f"  - {OUTPUT_DIR}/individual/   (individual outputs)")
    print(f"  - {OUTPUT_DIR}/spatial/      ")
    print(f"  - {OUTPUT_DIR}/appearance/   ")
    print(f"  - {OUTPUT_DIR}/combined/     ")
