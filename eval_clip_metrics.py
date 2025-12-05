"""
CLIP-based evaluation metrics for InstructPix2Pix.

Two metrics from the original paper:
1. CLIP Image Similarity: How well does the edit preserve the original image?
2. CLIP Directional Similarity: Does the edit match the instruction?
"""

import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path


class CLIPEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, image):
        """Encode image to CLIP embedding."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        embedding = self.model.encode_image(image_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        return embedding

    @torch.no_grad()
    def encode_text(self, text):
        """Encode text to CLIP embedding."""
        text_input = clip.tokenize([text], truncate=True).to(self.device)
        embedding = self.model.encode_text(text_input)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # Normalize
        return embedding

    def clip_image_similarity(self, input_image, output_image):
        """
        CLIP Image Similarity: How well does the edit preserve original content?

        Higher = better preservation of non-edited regions.
        Range: [-1, 1], typically [0.7, 1.0] for good edits.
        """
        input_emb = self.encode_image(input_image)
        output_emb = self.encode_image(output_image)

        similarity = (input_emb @ output_emb.T).item()
        return similarity

    def clip_directional_similarity(self, input_image, output_image, instruction):
        """
        CLIP Directional Similarity: Does the edit direction match the instruction?

        Measures if: (output_image - input_image) aligns with (instruction - "")
        Higher = edit better matches the instruction.
        Range: [-1, 1], typically [0.1, 0.3] for good edits.
        """
        # Image direction: how did the image change?
        input_emb = self.encode_image(input_image)
        output_emb = self.encode_image(output_image)
        image_direction = output_emb - input_emb
        image_direction = image_direction / image_direction.norm(dim=-1, keepdim=True)

        # Text direction: what change does the instruction describe?
        instruction_emb = self.encode_text(instruction)
        null_emb = self.encode_text("")
        text_direction = instruction_emb - null_emb
        text_direction = text_direction / text_direction.norm(dim=-1, keepdim=True)

        # Cosine similarity between directions
        similarity = (image_direction @ text_direction.T).item()
        return similarity

    def evaluate(self, input_image, output_image, instruction):
        """Compute both metrics."""
        img_sim = self.clip_image_similarity(input_image, output_image)
        dir_sim = self.clip_directional_similarity(input_image, output_image, instruction)
        return {
            "clip_image_similarity": img_sim,
            "clip_directional_similarity": dir_sim,
        }


def evaluate_folder(input_dir, output_dir, instructions):
    """
    Evaluate a folder of edited images.

    Args:
        input_dir: Folder with input images (input_1.png, input_2.png, ...)
        output_dir: Folder with output images (output_1.png, output_2.png, ...)
        instructions: List of edit instructions
    """
    evaluator = CLIPEvaluator()

    results = []
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    for i, instruction in enumerate(instructions):
        input_path = input_dir / f"input_{i+1}.png"
        output_path = output_dir / f"output_{i+1}.png"

        if input_path.exists() and output_path.exists():
            metrics = evaluator.evaluate(str(input_path), str(output_path), instruction)
            metrics["instruction"] = instruction
            results.append(metrics)
            print(f"{i+1}. '{instruction}'")
            print(f"   Image Sim: {metrics['clip_image_similarity']:.4f}")
            print(f"   Dir Sim:   {metrics['clip_directional_similarity']:.4f}")

    # Compute averages
    if results:
        avg_img_sim = np.mean([r["clip_image_similarity"] for r in results])
        avg_dir_sim = np.mean([r["clip_directional_similarity"] for r in results])
        print(f"\n--- Averages ---")
        print(f"CLIP Image Similarity:       {avg_img_sim:.4f}")
        print(f"CLIP Directional Similarity: {avg_dir_sim:.4f}")

    return results


if __name__ == "__main__":
    # Example usage with your test outputs
    evaluator = CLIPEvaluator()

    # Spatial test prompts (same as test_lora_simple.py)
    prompts = [
        "Put a hat on the cat's head",
        "Add sunglasses on the cat's face",
        "Place a ball in front of the cat",
        "Add a bird flying behind the cat",
        "Turn this cat into a dog",
    ]

    # Models to compare: (folder_name, display_name)
    experiments = [
        ("instruct-pix2pix-00-22000.ckpt", "Baseline InstructPix2Pix"),
        ("train_train_spatial_full_10ep", "Full Finetune on Spatial Data (10 epochs)"),
        ("train_spatial_full_finetune_2ep", "Full Finetune on Spatial Data (2 epochs)"),
        ("train_spatial_lora_test", "LoRA on Spatial Data"),
        ("train_hq_edit_lora_rank32_100files_50epochs", "LoRA on HQ-Edit (rank=32)"),
        ("train_normal_lora_r8_lr5e5_1ep", "LoRA on HQ-Edit (rank=8, 1 epoch)"),
        ("train_depth_conditioned_depth_cond_5ep", "Depth Conditioned (5 epochs)"),
    ]

    # Store results for summary table
    summary = []

    for exp_name, display_name in experiments:
        results_dir = f"{exp_name}_results"
        input_image = f"{results_dir}/{exp_name}_input.png"

        # Check if results exist
        import os
        if not os.path.exists(results_dir):
            print(f"\n[SKIP] {display_name} - results not found")
            continue

        print("\n" + "=" * 70)
        print(f"MODEL: {display_name}")
        print("=" * 70)

        test_cases = [
            (f"{results_dir}/{exp_name}_output_{i+1}.png", prompts[i])
            for i in range(len(prompts))
        ]

        all_img_sim = []
        all_dir_sim = []

        for output_image, instruction in test_cases:
            try:
                metrics = evaluator.evaluate(input_image, output_image, instruction)
                print(f"\n  '{instruction}'")
                print(f"    CLIP-I: {metrics['clip_image_similarity']:.4f}  |  CLIP-D: {metrics['clip_directional_similarity']:.4f}")
                all_img_sim.append(metrics['clip_image_similarity'])
                all_dir_sim.append(metrics['clip_directional_similarity'])
            except Exception as e:
                print(f"  Error: '{instruction}': {e}")

        if all_img_sim:
            avg_i = np.mean(all_img_sim)
            avg_d = np.mean(all_dir_sim)
            print(f"\n  >>> AVERAGE: CLIP-I = {avg_i:.4f}  |  CLIP-D = {avg_d:.4f}")
            summary.append((display_name, avg_i, avg_d))

    # Print summary table
    print("SUMMARY TABLE")
    print(f"{'Model':<45} {'CLIP-I':>10} {'CLIP-D':>10}")
    for name, clip_i, clip_d in summary:
        print(f"{name:<45} {clip_i:>10.4f} {clip_d:>10.4f}")

    print("INTERPRETATION")
    print("  CLIP-I (Image Similarity):  Higher = better preservation (~0.85-0.95 good)")
    print("  CLIP-D (Directional Sim):   Higher = edit matches instruction (~0.15-0.30 good)")
    print("  Trade-off: High CLIP-I + High CLIP-D = ideal edit")
