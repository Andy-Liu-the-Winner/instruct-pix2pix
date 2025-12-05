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



import io
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset


class AuroraEditDataset(Dataset):
    """
    Dataset wrapper for the McGill-NLP/AURORA dataset for InstructPix2Pix-style training.

    Expected HF dataset format (per example):
      - 'input'       : before image (source)
      - 'output'      : after image  (target)
      - 'instruction' : text edit prompt

    Returns examples in the format compatible with ddpm_edit.LatentDiffusion:

      {
          "edited": output_image,        # Target (what we want to generate)
          "edit": {
              "c_concat": input_image,   # Condition: input image
              "c_crossattn": prompt      # Condition: text instruction
          }
      }

    Where input_image and output_image are float32 numpy arrays
    in CHW format, normalized to [-1, 1].
    """

    def __init__(
        self,
        hf_dataset_name: str = "McGill-NLP/AURORA",
        split: str = "train",
        crop_res: int = 256,
    ):
        super().__init__()
        self.crop_res = crop_res

        print(f"[AuroraEditDataset] Loading dataset '{hf_dataset_name}', split='{split}' from HF hub...")
        self.dataset = load_dataset(hf_dataset_name, split=split)
        print(f"[AuroraEditDataset] Loaded {len(self.dataset)} examples.")

        if len(self.dataset) > 0:
            print("[AuroraEditDataset] Example keys:", self.dataset[0].keys())

    def __len__(self):
        return len(self.dataset)

    def _to_pil(self, img):
        """Ensure the image is a RGB PIL.Image."""
        if isinstance(img, Image.Image):
            pil = img
        elif isinstance(img, bytes):
            pil = Image.open(io.BytesIO(img)).convert("RGB")
        elif isinstance(img, dict) and "bytes" in img:
            pil = Image.open(io.BytesIO(img["bytes"])).convert("RGB")
        else:
            pil = Image.fromarray(np.array(img)).convert("RGB")

        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return pil

    def __getitem__(self, i):
        ex = self.dataset[i]

        # AURORA columns
        input_image  = ex.get("input")
        output_image = ex.get("output")
        prompt       = ex.get("instruction", "")

        # to PIL
        input_image  = self._to_pil(input_image)
        output_image = self._to_pil(output_image)

        # resize to (crop_res, crop_res)
        input_image  = input_image.resize((self.crop_res, self.crop_res), Image.LANCZOS)
        output_image = output_image.resize((self.crop_res, self.crop_res), Image.LANCZOS)

        # to numpy in [-1, 1]
        input_np  = np.array(input_image).astype(np.float32) / 127.5 - 1.0
        output_np = np.array(output_image).astype(np.float32) / 127.5 - 1.0

        # HWC -> CHW
        input_np  = input_np.transpose(2, 0, 1)
        output_np = output_np.transpose(2, 0, 1)

        return {
            "edited": output_np,      # first_stage_key in YAML
            "edit": {
                "c_concat": input_np, # cond image
                "c_crossattn": prompt,
            }
        }
