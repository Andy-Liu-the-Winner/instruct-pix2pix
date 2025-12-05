"""
Automated evaluation script: Generate outputs for all models and compute CLIP metrics.
Runs inference on multiple test images and computes average CLIP-I and CLIP-D scores.
"""

import sys
sys.path.append("./stable_diffusion")

import torch
import clip
from PIL import Image
import requests
from io import BytesIO
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np
import os
from pathlib import Path
import json
from PIL import ImageDraw, ImageFont

# ============== CONFIG ==============

# Test images (URL, name)
TEST_IMAGES = [
    # Animals
    ("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png", "cat"),
    ("https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=512", "cat_orange"),
    ("https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=512", "dog"),
    ("https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=512", "dog_golden"),
    ("https://images.unsplash.com/photo-1474511320723-9a56873571b7?w=512", "horse"),
    # People
    ("https://raw.githubusercontent.com/timothybrooks/instruct-pix2pix/main/imgs/example.jpg", "girl"),
    ("https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=512", "man_portrait"),
    ("https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=512", "woman_portrait"),
    # Scenes
    ("https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=512", "mountain"),
    ("https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=512", "beach"),
    ("https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=512", "city"),
    ("https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=512", "forest"),
    # Objects
    ("https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=512", "watch"),
    ("https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=512", "headphones"),
    ("https://images.unsplash.com/photo-1526170375885-4d8ecf77b99f?w=512", "camera"),
]

# Test prompts (spatial-focused)
TEST_PROMPTS = [
    "Put a hat on the subject's head",
    "Add sunglasses on the face",
    "Place a ball in front of the subject",
    "Add a bird flying in the background",
    "Move the subject to the left side",
]

# Models to evaluate: (checkpoint_path, config_path, display_name)
MODELS = [
    # ("checkpoints/instruct-pix2pix-00-22000.ckpt", "configs/train.yaml", "Baseline InstructPix2Pix"),
    # ("logs/train_spatial_lora_test/checkpoints/last.ckpt", "configs/train.yaml", "LoRA on Spatial Data"),
    # ("logs/train_train_spatial_full_10ep/checkpoints/last.ckpt", "configs/train.yaml", "Full Finetune Spatial (10ep)"),
    # ("logs/train_spatial_full_finetune_2ep/checkpoints/last.ckpt", "configs/train.yaml", "Full Finetune Spatial (2ep)"),
    # ("logs/train_hq_edit_lora_rank32_100files_50epochs/checkpoints/last.ckpt", "configs/train.yaml", "LoRA on HQ-Edit (r=32)"),
    # ("logs/train_esplora_train_esplora/checkpoints/last.ckpt", "configs/train_esplora.yaml", "ESPLoRA (r=64)"),
    ("logs/train_esplora_r128_train_esplora_r128/checkpoints/last.ckpt", "configs/train_esplora_r128.yaml", "ESPLoRA (r=128)"),
]

STEPS = 50  # Fewer steps for faster evaluation
OUTPUT_DIR = "eval_results"
SAVE_IMAGES = True  # Save generated images

# ============== CLIP EVALUATOR ==============

class CLIPEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(image_input)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, text):
        text_input = clip.tokenize([text], truncate=True).to(self.device)
        embedding = self.model.encode_text(text_input)
        return embedding / embedding.norm(dim=-1, keepdim=True)

    def clip_image_similarity(self, input_image, output_image):
        input_emb = self.encode_image(input_image)
        output_emb = self.encode_image(output_image)
        return (input_emb @ output_emb.T).item()

    def clip_directional_similarity(self, input_image, output_image, instruction):
        input_emb = self.encode_image(input_image)
        output_emb = self.encode_image(output_image)
        image_direction = output_emb - input_emb
        image_direction = image_direction / image_direction.norm(dim=-1, keepdim=True)

        instruction_emb = self.encode_text(instruction)
        null_emb = self.encode_text("")
        text_direction = instruction_emb - null_emb
        text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)

        return (image_direction @ text_direction.T).item()


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
def edit_image(model, input_image, prompt, steps=50):
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

        # Handle depth conditioning if enabled
        if hasattr(model, 'use_depth_conditioning') and model.use_depth_conditioning:
            if model.depth_conditioner is not None:
                depth_3ch = model.depth_conditioner.get_depth(image_tensor)
                if depth_3ch is not None:
                    depth_3ch = depth_3ch.cuda()
                    depth_latent = model.encode_first_stage(depth_3ch).mode()
                    cond["c_concat"].append(depth_latent)
                    uncond["c_concat"].append(torch.zeros_like(depth_latent))

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

    # Initialize CLIP evaluator
    print("Loading CLIP model...")
    evaluator = CLIPEvaluator()

    # Load test images
    print("\nLoading test images...")
    test_images = []
    for url, name in TEST_IMAGES:
        try:
            img = load_image_from_url(url)
            test_images.append((img, name))
            print(f"  Loaded: {name}")
        except Exception as e:
            print(f"  Failed to load {name}: {e}")

    # Results storage - load existing results if available
    results_path = os.path.join(OUTPUT_DIR, "clip_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            all_results = json.load(f)
        print(f"Loaded existing results for {len(all_results)} models from {results_path}")
    else:
        all_results = {}

    # Evaluate each model
    for ckpt_path, config_path, model_name in MODELS:
        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}")
        print(f"{'='*70}")

        # Check if checkpoint exists
        if not os.path.exists(ckpt_path):
            print(f"  [SKIP] Checkpoint not found: {ckpt_path}")
            continue

        # Check if already evaluated (results folder exists with images)
        model_folder = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        model_output_dir = os.path.join(OUTPUT_DIR, model_folder)
        if os.path.exists(model_output_dir) and len(os.listdir(model_output_dir)) > 10:
            print(f"  [SKIP] Already evaluated, results exist in: {model_output_dir}")
            continue

        # Load model
        print(f"  Loading checkpoint: {ckpt_path}")
        try:
            model = load_model(ckpt_path, config_path)
        except Exception as e:
            print(f"  [ERROR] Failed to load model: {e}")
            continue

        model_results = {"clip_i": [], "clip_d": []}

        # Create model output folder
        model_folder = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        model_output_dir = os.path.join(OUTPUT_DIR, model_folder)
        if SAVE_IMAGES:
            os.makedirs(model_output_dir, exist_ok=True)

        # Test on each image
        for input_image, img_name in test_images:
            print(f"\n  Image: {img_name}")

            # Save input image
            if SAVE_IMAGES:
                input_image.save(os.path.join(model_output_dir, f"{img_name}_input.png"))

            for prompt_idx, prompt in enumerate(TEST_PROMPTS):
                try:
                    # Generate edited image
                    output_image = edit_image(model, input_image.copy(), prompt, steps=STEPS)

                    # Save output image
                    if SAVE_IMAGES:
                        safe_prompt = prompt[:30].replace(" ", "_").replace("'", "")
                        output_image.save(os.path.join(model_output_dir, f"{img_name}_p{prompt_idx+1}_{safe_prompt}.png"))

                    # Compute CLIP metrics
                    clip_i = evaluator.clip_image_similarity(input_image, output_image)
                    clip_d = evaluator.clip_directional_similarity(input_image, output_image, prompt)

                    model_results["clip_i"].append(clip_i)
                    model_results["clip_d"].append(clip_d)

                    print(f"    [{prompt_idx+1}] CLIP-I: {clip_i:.4f} | CLIP-D: {clip_d:.4f} | {prompt[:30]}...")

                except Exception as e:
                    print(f"    [{prompt_idx+1}] ERROR: {e}")

        # Compute averages
        if model_results["clip_i"]:
            avg_clip_i = np.mean(model_results["clip_i"])
            avg_clip_d = np.mean(model_results["clip_d"])
            std_clip_i = np.std(model_results["clip_i"])
            std_clip_d = np.std(model_results["clip_d"])

            all_results[model_name] = {
                "clip_i_mean": avg_clip_i,
                "clip_i_std": std_clip_i,
                "clip_d_mean": avg_clip_d,
                "clip_d_std": std_clip_d,
                "n_samples": len(model_results["clip_i"])
            }

            print(f"\n  >>> AVERAGE: CLIP-I = {avg_clip_i:.4f} ± {std_clip_i:.4f} | CLIP-D = {avg_clip_d:.4f} ± {std_clip_d:.4f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<40} {'CLIP-I':>12} {'CLIP-D':>12} {'N':>6}")
    print("-"*70)
    for model_name, results in all_results.items():
        clip_i_str = f"{results['clip_i_mean']:.4f}±{results['clip_i_std']:.3f}"
        clip_d_str = f"{results['clip_d_mean']:.4f}±{results['clip_d_std']:.3f}"
        print(f"{model_name:<40} {clip_i_str:>12} {clip_d_str:>12} {results['n_samples']:>6}")

    # Save results to JSON
    results_path = os.path.join(OUTPUT_DIR, "clip_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ============== GENERATE COMPARISON GRID ==============
    if SAVE_IMAGES:
        print("\nGenerating comparison grids...")

        # Get list of model folders that were evaluated
        model_folders = []
        model_display_names = []
        for ckpt_path, config_path, model_name in MODELS:
            if os.path.exists(ckpt_path):
                folder = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
                if os.path.exists(os.path.join(OUTPUT_DIR, folder)):
                    model_folders.append(folder)
                    model_display_names.append(model_name)

        # Create grid for each test image
        for img_url, img_name in TEST_IMAGES:
            try:
                # Grid layout: rows = prompts, cols = models
                # Plus header row and input column
                cell_size = 256
                header_height = 40
                label_width = 200

                n_models = len(model_folders)
                n_prompts = len(TEST_PROMPTS)

                grid_width = label_width + (n_models + 1) * cell_size  # +1 for input column
                grid_height = header_height + n_prompts * cell_size

                grid = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))
                draw = ImageDraw.Draw(grid)

                # Try to load a font, fall back to default
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
                    font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
                except:
                    font = ImageFont.load_default()
                    font_small = font

                # Draw header (model names)
                draw.text((label_width + cell_size//2 - 20, 10), "Input", fill=(0, 0, 0), font=font)
                for col, name in enumerate(model_display_names):
                    x = label_width + (col + 2) * cell_size - cell_size//2 - len(name)*3
                    # Truncate long names
                    short_name = name[:25] + "..." if len(name) > 25 else name
                    draw.text((x, 10), short_name, fill=(0, 0, 0), font=font_small)

                # Load input image
                input_path = os.path.join(OUTPUT_DIR, model_folders[0], f"{img_name}_input.png")
                if os.path.exists(input_path):
                    input_img = Image.open(input_path).resize((cell_size, cell_size), Image.LANCZOS)
                else:
                    input_img = Image.new("RGB", (cell_size, cell_size), (200, 200, 200))

                # Fill grid
                for row, prompt in enumerate(TEST_PROMPTS):
                    y = header_height + row * cell_size

                    # Draw prompt label (truncated)
                    short_prompt = prompt[:28] + "..." if len(prompt) > 28 else prompt
                    draw.text((5, y + cell_size//2 - 10), short_prompt, fill=(0, 0, 0), font=font_small)

                    # Paste input image (same for all rows)
                    grid.paste(input_img, (label_width, y))

                    # Paste each model's output
                    for col, folder in enumerate(model_folders):
                        safe_prompt = prompt[:30].replace(" ", "_").replace("'", "")
                        output_path = os.path.join(OUTPUT_DIR, folder, f"{img_name}_p{row+1}_{safe_prompt}.png")

                        if os.path.exists(output_path):
                            output_img = Image.open(output_path).resize((cell_size, cell_size), Image.LANCZOS)
                        else:
                            output_img = Image.new("RGB", (cell_size, cell_size), (200, 200, 200))

                        x = label_width + (col + 1) * cell_size
                        grid.paste(output_img, (x, y))

                # Save grid
                grid_path = os.path.join(OUTPUT_DIR, f"comparison_grid_{img_name}.png")
                grid.save(grid_path)
                print(f"  Saved: {grid_path}")

            except Exception as e:
                print(f"  Error creating grid for {img_name}: {e}")

        # Create summary grid (one image per model, best example)
        print("\nGenerating summary grid (all models, sample outputs)...")
        try:
            cell_size = 256
            header_height = 50
            n_models = len(model_folders)
            n_samples = min(5, len(TEST_IMAGES))  # Show 5 sample images

            summary_width = (n_models + 1) * cell_size
            summary_height = header_height + n_samples * cell_size

            summary = Image.new("RGB", (summary_width, summary_height), (255, 255, 255))
            draw = ImageDraw.Draw(summary)

            # Header
            draw.text((cell_size//2 - 20, 15), "Input", fill=(0, 0, 0), font=font)
            for col, name in enumerate(model_display_names):
                x = (col + 1) * cell_size + cell_size//4
                short_name = name[:20] if len(name) <= 20 else name[:17] + "..."
                draw.text((x, 15), short_name, fill=(0, 0, 0), font=font_small)

            # Sample images (first prompt only)
            for row, (img_url, img_name) in enumerate(TEST_IMAGES[:n_samples]):
                y = header_height + row * cell_size

                # Input
                input_path = os.path.join(OUTPUT_DIR, model_folders[0], f"{img_name}_input.png")
                if os.path.exists(input_path):
                    img = Image.open(input_path).resize((cell_size, cell_size), Image.LANCZOS)
                    summary.paste(img, (0, y))

                # Each model's output (prompt 1)
                for col, folder in enumerate(model_folders):
                    safe_prompt = TEST_PROMPTS[0][:30].replace(" ", "_").replace("'", "")
                    output_path = os.path.join(OUTPUT_DIR, folder, f"{img_name}_p1_{safe_prompt}.png")

                    if os.path.exists(output_path):
                        img = Image.open(output_path).resize((cell_size, cell_size), Image.LANCZOS)
                        summary.paste(img, ((col + 1) * cell_size, y))

            summary_path = os.path.join(OUTPUT_DIR, "summary_grid.png")
            summary.save(summary_path)
            print(f"  Saved: {summary_path}")

        except Exception as e:
            print(f"  Error creating summary grid: {e}")

    print("INTERPRETATION")
    print("="*80)
    print("  CLIP-I (Image Similarity):  Higher = better preservation (~0.85-0.95 good)")
    print("  CLIP-D (Directional Sim):   Higher = edit matches instruction (~0.15-0.30 good)")
