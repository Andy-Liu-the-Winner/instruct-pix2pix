# data from https://huggingface.co/datasets/UCSC-VLAA/HQ-Editw
# This is a PyTorch Dataset loader for the HQ-Edit image editing dataset. It:

#   1. Loads data from local cache (no internet downloads during training)
#      Finds parquet files in ~/.cache/huggingface/hub/datasets--UCSC-VLAA--HQ-Edit/, you might need to download the dataset manually first
#      Currently loads 100 files (~40,000-60,000 examples) for faster training time
#   2. Provides training data in triplets:
#      input_image = before image (e.g., a cat photo)
#      edit = text instruction (e.g., "turn this cat into a dog")
#      output_image = after image (e.g., a dog photo)
#   3. Applies data augmentation:
#      Random resize
#      Random crop
#      Random horizontal flip
#      Normalize to [-1, 1]
#   4. Returns data in InstructPix2Pix format:
#   {
#       "edited": output_image,        # Target (what model should generate)
#       "edit": {
#           "c_concat": input_image,   # Condition: input image
#           "c_crossattn": prompt      # Condition: text instruction
#       }
#   }
# tested 100 files takes around 30 mins to load examples (on A100) and around 13 mins to train one epoch





import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
import io


class HFEditDataset(Dataset):
    """
    Wrapper for HuggingFace datasets like HQ-Edit for InstructPix2Pix training.

    Expected dataset format:
    - 'input': PIL Image (before image)
    - 'output': PIL Image (after image)
    - 'instruction' or 'edit': text prompt
    """

    def __init__(
        self,
        hf_dataset_name="UCSC-VLAA/HQ-Edit",
        split="train",
        min_resize_res=256,
        max_resize_res=256,
        crop_res=256,
        flip_prob=0.5,
    ):
        self.split = split
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        print(f"Loading HuggingFace dataset from CACHE ONLY: {hf_dataset_name}, split: {split}")

        # Load directly from cached parquet files - NO DOWNLOADS!
        import os
        import glob

        # Find all cached parquet files
        parquet_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--UCSC-VLAA--HQ-Edit/snapshots/*/data/*.parquet")
        parquet_files = glob.glob(parquet_pattern)

        if not parquet_files:
            raise RuntimeError(f"No cached parquet files found at {parquet_pattern}")

        print(f"Found {len(parquet_files)} cached parquet files")

        # TEST MODE: Use only first 5 files for quick testing (~2000-3000 examples)
        # Comment out this line to use all files
        # parquet_files = parquet_files[:5] experiment 1
        parquet_files = parquet_files[:100]
        print(f"TEST MODE: Using only {len(parquet_files)} files for quick validation")

        # Load directly from parquet files - NO INTERNET ACCESS
        self.dataset = load_dataset("parquet", data_files=parquet_files, split=split)

        print(f"Loaded {len(self.dataset)} examples from LOCAL CACHE - NO DOWNLOADS!")

        # Print first example to understand structure
        if len(self.dataset) > 0:
            print("\nDataset structure (first example keys):")
            print(self.dataset[0].keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        example = self.dataset[i]

        # Extract images and prompt
        # HQ-Edit dataset keys: 'input_image', 'output_image', 'edit'
        input_image = example.get('input_image')
        output_image = example.get('output_image')
        prompt = example.get('edit') or example.get('instruction') or ""

        # Convert to PIL if needed
        if not isinstance(input_image, Image.Image):
            if isinstance(input_image, bytes):
                input_image = Image.open(io.BytesIO(input_image)).convert("RGB")
            elif isinstance(input_image, dict) and 'bytes' in input_image:
                input_image = Image.open(io.BytesIO(input_image['bytes'])).convert("RGB")
            else:
                input_image = Image.fromarray(input_image).convert("RGB")

        if not isinstance(output_image, Image.Image):
            if isinstance(output_image, bytes):
                output_image = Image.open(io.BytesIO(output_image)).convert("RGB")
            elif isinstance(output_image, dict) and 'bytes' in output_image:
                output_image = Image.open(io.BytesIO(output_image['bytes'])).convert("RGB")
            else:
                output_image = Image.fromarray(output_image).convert("RGB")

        # Apply transformations (matching InstructPix2Pix preprocessing)
        reize_res = np.random.randint(self.min_resize_res, self.max_resize_res + 1)
        input_image = input_image.resize((reize_res, reize_res), Image.LANCZOS)
        output_image = output_image.resize((reize_res, reize_res), Image.LANCZOS)

        # Random crop
        if reize_res > self.crop_res:
            input_image = self._random_crop(input_image, self.crop_res)
            output_image = self._random_crop(output_image, self.crop_res)

        # Random horizontal flip
        if np.random.rand() < self.flip_prob:
            input_image = input_image.transpose(Image.FLIP_LEFT_RIGHT)
            output_image = output_image.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to numpy and normalize to [-1, 1]
        input_image = np.array(input_image).astype(np.float32) / 127.5 - 1.0
        output_image = np.array(output_image).astype(np.float32) / 127.5 - 1.0

        # Transpose to CHW format
        input_image = input_image.transpose(2, 0, 1)
        output_image = output_image.transpose(2, 0, 1)

        return {
            "edited": output_image,  # Target image (what we want to generate)
            "edit": {
                "c_concat": input_image,  # Input image (condition)
                "c_crossattn": prompt,     # Text instruction (condition)
            }
        }

    def _random_crop(self, image, crop_res):
        """Random crop to crop_res x crop_res"""
        # If image is 256x256 and crop_res is 256, crop the full image
        # If image is 300x300 and crop_res is 256, crop a random 256x256 region
        width, height = image.size
        left = np.random.randint(0, width - crop_res + 1)
        top = np.random.randint(0, height - crop_res + 1)
        return image.crop((left, top, left + crop_res, top + crop_res))