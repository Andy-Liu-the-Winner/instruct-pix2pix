"""
Generate images with a single model checkpoint.
Quick script for generating results from one model.
"""

import sys
sys.path.append("./stable_diffusion")

import torch
import requests
from PIL import Image
from io import BytesIO
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np
import os

# ============== CONFIG ==============

# Model checkpoint
MODEL_CKPT = "logs/train_esplora_r128_train_esplora_r128/checkpoints/last.ckpt"
MODEL_CONFIG = "configs/train_esplora_r128.yaml"
MODEL_NAME = "ESPLoRA_r128"

# Output directory
OUTPUT_DIR = f"results_{MODEL_NAME}"

# Test images
TEST_IMAGES = [
    ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "cat"),
    ("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512", "man"),
    ("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512", "woman"),
    ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512", "mountain"),
]

# Test prompts
TEST_PROMPTS = [
    "Move the subject to the left side",
    "Move the subject to the right side",
    "Add sunglasses",
    "Put a hat on the subject",
    "Make it look like a painting",
    "Change to winter with snow",
    "Place a ball to the left of the subject",
    "Add a bird above the subject",
]

STEPS = 100

# ============== HELPERS ==============

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

# ============== MAIN ==============

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Model: {MODEL_NAME}")
    print(f"Checkpoint: {MODEL_CKPT}")
    print(f"Output: {OUTPUT_DIR}/\n")

    # Load model
    print("Loading model...")
    model = load_model(MODEL_CKPT, MODEL_CONFIG)
    print("✓ Model loaded!\n")

    # Load test images
    print("Loading test images...")
    test_images = []
    for url, name in TEST_IMAGES:
        try:
            img = load_image_from_url(url)
            test_images.append((img, name))
            # Save input
            img.save(os.path.join(OUTPUT_DIR, f"{name}_input.png"))
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    # Generate
    total = len(test_images) * len(TEST_PROMPTS)
    print(f"\nGenerating {total} images ({len(test_images)} images × {len(TEST_PROMPTS)} prompts)...")
    print(f"Estimated time: ~{total * 5 / 60:.1f} minutes\n")

    count = 0
    for img, img_name in test_images:
        for prompt_idx, prompt in enumerate(TEST_PROMPTS):
            count += 1
            print(f"[{count}/{total}] {img_name}: {prompt[:40]}...")

            try:
                output = edit_image(model, img.copy(), prompt, steps=STEPS)

                safe_prompt = prompt[:30].replace(" ", "_")
                output_path = os.path.join(OUTPUT_DIR, f"{img_name}_p{prompt_idx:02d}_{safe_prompt}.png")
                output.save(output_path)

                print(f"  ✓ Saved: {output_path}")

            except Exception as e:
                print(f"  ✗ ERROR: {e}")

    print(f"\n{'='*70}")
    print(f"✓ DONE!")
    print(f"{'='*70}")
    print(f"Generated {total} images")
    print(f"Results saved to: {OUTPUT_DIR}/")
