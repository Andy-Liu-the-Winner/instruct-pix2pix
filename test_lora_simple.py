import sys
sys.path.append("./stable_diffusion")

import torch
from PIL import Image, ImageOps
import requests
from io import BytesIO
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np
import os
import math
from pathlib import Path

def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_model(checkpoint_path):
    config = OmegaConf.load("configs/train.yaml")
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

    # Run inference
    with autocast("cuda"):
        # Maintain aspect ratio, pad to multiples of 64 (no cropping!)
        width, height = input_image.size
        resolution = 512  # Use 512 like HQ-Edit

        # Resize to fit within resolution while maintaining aspect ratio
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

        # Pad to multiples of 64
        pad_width = ((new_width + 63) // 64) * 64
        pad_height = ((new_height + 63) // 64) * 64

        if pad_width != new_width or pad_height != new_height:
            padded = Image.new("RGB", (pad_width, pad_height), (0, 0, 0))
            padded.paste(input_image, ((pad_width - new_width) // 2, (pad_height - new_height) // 2))
            input_image = padded

        image_tensor = 2 * torch.tensor(np.array(input_image)).float() / 255 - 1
        image_tensor = rearrange(image_tensor, "h w c -> 1 c h w").cuda()

        cond = {
            "c_crossattn": [model.get_learned_conditioning([prompt])],
            "c_concat": [model.encode_first_stage(image_tensor).mode()]
        }

        null_token = model.get_learned_conditioning([""])
        uncond = {
            "c_crossattn": [null_token],
            "c_concat": [torch.zeros_like(cond["c_concat"][0])]
        }

        model_wrap = K.external.CompVisDenoiser(model)

        class CFGDenoiser:
            def __init__(self, inner_model):
                self.inner_model = inner_model

            def __call__(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
                cfg_z = repeat(z, "1 ... -> n ...", n=3)
                cfg_sigma = repeat(sigma, "1 ... -> n ...", n=3)
                cfg_cond = {
                    "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
                    "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
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

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    checkpoint_path = "logs/train_spatial_spatial_lora_on_ip2p/checkpoints/last.ckpt"
    model = load_model(checkpoint_path)

    # Extract experiment name from checkpoint path
    experiment_name = Path(checkpoint_path).parts[1]
    output_dir = f"{experiment_name}_results"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    # Load test image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    input_image = load_image(image_url)
    input_image.save(f"{output_dir}/{experiment_name}_input.png")

    # Test prompts
    prompts = [
        "do not edit anything here",
        "Turn this cat into a dog",
        "Make a dog out of the cat",
        "Turn the cat wear a hat, do not change the background",
        "Turn the cat look like a lion, but it should still be a cat",
    ]

    # Generate
    print("Generating images...")
    for i, prompt in enumerate(prompts):
        print(f"  {i+1}. {prompt}")
        output = edit_image(model, input_image, prompt, steps=100)
        output.save(f"{output_dir}/{experiment_name}_output_{i+1}.png")

    print(f"Done! Results saved to {output_dir}/")
