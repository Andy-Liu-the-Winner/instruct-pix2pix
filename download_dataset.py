"""
Download HQ-Edit dataset to local cache.
Run this once before training.
"""
from datasets import load_dataset

print("Downloading HQ-Edit dataset...")
print("This will download to ~/.cache/huggingface/hub/")
print("Dataset size: ~30-40GB")
print()

# Download the full dataset
dataset = load_dataset("UCSC-VLAA/HQ-Edit", split="train")

print(f"\nDownload complete!")
print(f"Total examples: {len(dataset)}")
print(f"Cached at: ~/.cache/huggingface/hub/datasets--UCSC-VLAA--HQ-Edit/")
print()
print("You can now run training with:")
print("  python main.py --name my_experiment --base configs/train.yaml --train --gpus 0,")
