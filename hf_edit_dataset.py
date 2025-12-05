# data from https://huggingface.co/datasets/UCSC-VLAA/HQ-Editw
# This is a PyTorch Dataset loader for the HQ-Edit image editing dataset. It:

#   1. Loads data from local cache (no internet downloads during training)
#      Finds parquet files in ~/.cache/huggingface/hub/datasets--UCSC-VLAA--HQ-Edit/, you might need to download the dataset manually first
#      Currently loads 100 files (~40,000-60,000 examples) for faster training time
#   2. Provides training data in triplets:
#      input_image = before image (e.g., a cat photo)
#      edit = text instruction (e.g., "turn this cat into a dog")
#      output_image = after image (e.g., a dog photo)
#   3. Applies data augmentation (before midway report):
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
from datasets import load_dataset, load_from_disk
import io
import os
import glob


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
        saved_dataset_path=None,  # NEW: Load from saved filtered dataset
    ):
        self.split = split
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob

        # Load from saved filtered dataset (e.g., spatial_edits_hq_only)
        if saved_dataset_path:
            print(f"Loading from SAVED dataset: {saved_dataset_path}")

            # Check if it's a sharded dataset (multiple shard_XXXX directories)
            shard_pattern = os.path.join(saved_dataset_path, "shard_*")
            shard_dirs = sorted(glob.glob(shard_pattern))

            if shard_dirs:
                # Load and concatenate all shards
                print(f"Found {len(shard_dirs)} shards, loading...")
                from datasets import concatenate_datasets
                datasets = []
                skipped = []
                for shard_dir in shard_dirs:
                    try:
                        ds = load_from_disk(shard_dir)
                        datasets.append(ds)
                        print(f"  Loaded {shard_dir}: {len(ds)} examples")
                    except Exception as e:
                        print(f"  SKIPPED {shard_dir} (corrupted): {e}")
                        skipped.append(shard_dir)
                if skipped:
                    print(f"WARNING: {len(skipped)} shards corrupted, using {len(datasets)} shards")
                self.dataset = concatenate_datasets(datasets)
                print(f"Total: {len(self.dataset)} examples from {len(datasets)} shards")
            else:
                # Single dataset directory (old format)
                self.dataset = load_from_disk(saved_dataset_path)
                print(f"Loaded {len(self.dataset)} examples from saved dataset")
        else:
            # Otherwise, Load from parquet cache (original behavior)
            print(f"Loading HuggingFace dataset from CACHE ONLY: {hf_dataset_name}, split: {split}")

            parquet_pattern = os.path.expanduser("~/.cache/huggingface/hub/datasets--UCSC-VLAA--HQ-Edit/snapshots/*/data/*.parquet")
            parquet_files = glob.glob(parquet_pattern)

            if not parquet_files:
                raise RuntimeError(f"No cached parquet files found at {parquet_pattern}")

            print(f"Found {len(parquet_files)} cached parquet files")

            # TEST MODE: Use only first 100 files
            parquet_files = parquet_files[:100]
            print(f"TEST MODE: Using only {len(parquet_files)} files for quick validation")

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

        # Resize to target resolution (no random crop or flip - cleaner for spatial learning)
        input_image = input_image.resize((self.crop_res, self.crop_res), Image.LANCZOS)
        output_image = output_image.resize((self.crop_res, self.crop_res), Image.LANCZOS)

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

