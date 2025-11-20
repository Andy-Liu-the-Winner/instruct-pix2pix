import sys
sys.path.append("./stable_diffusion")

import torch
from PIL import Image
import requests
from io import BytesIO
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import numpy as np

def load_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

def load_model_from_checkpoint(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model = model.cuda()
    model.eval()
    return model

def preprocess_image(image, size=256):
    image = image.resize((size, size), Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 127.5 - 1.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

def postprocess_image(tensor):
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    image = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)

@torch.no_grad()
def edit_image(model, image, prompt, num_steps=20):
    image_tensor = preprocess_image(image).cuda()

    encoder_posterior = model.encode_first_stage(image_tensor)
    z = model.get_first_stage_encoding(encoder_posterior).detach()

    c = model.get_learned_conditioning([prompt])

    cond = {"c_concat": [z], "c_crossattn": [c]}

    shape = [4, 256//8, 256//8]
    samples, _ = model.sample(cond=cond, batch_size=1, shape=shape,
                              verbose=False, unconditional_guidance_scale=7.5,
                              unconditional_conditioning=None, eta=1.0)

    x_samples = model.decode_first_stage(samples)
    return postprocess_image(x_samples)

if __name__ == "__main__":
    # Load test image
    image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
    input_image = load_image(image_url)
    input_image.save("test_input.png")

    prompt = "Turn this cat into a dog"

    # Test base model
    print("Loading base model...")
    base_model = load_model_from_checkpoint(
        "configs/train.yaml",
        "stable_diffusion/models/ldm/stable-diffusion-v1/v1-5-pruned-emaonly.ckpt"
    )
    print("Running base model...")
    output_base = edit_image(base_model, input_image, prompt)
    output_base.save("test_output_base.png")
    del base_model
    torch.cuda.empty_cache()

    # Test LoRA model
    print("Loading LoRA model...")
    lora_model = load_model_from_checkpoint(
        "configs/train.yaml",
        "logs/train_hq_edit_lora_experiment_optimizer_fixed_trainable_for_lora/checkpoints/epoch=000019.ckpt"
    )
    print("Running LoRA model...")
    output_lora = edit_image(lora_model, input_image, prompt)
    output_lora.save("test_output_lora.png")

    print("Done! Compare outputs:")
    print("  test_input.png - original")
    print("  test_output_base.png - base model")
    print("  test_output_lora.png - LoRA model")
