# check_spatial_dataset.py
# Quick script to verify the saved spatial dataset

from datasets import load_from_disk
import os

def main():
    dataset_path = "spatial_edits_hq_only"

    if not os.path.exists(dataset_path):
        print(f"ERROR: Dataset not found at {dataset_path}")
        return

    print(f"Loading dataset from {dataset_path}...")
    ds = load_from_disk(dataset_path)

    print(f"\n{'='*50}")
    print(f"Dataset loaded successfully!")
    print(f"Total examples: {len(ds)}")
    print(f"Columns: {ds.column_names}")
    print(f"{'='*50}")

    # Show sample instructions
    print("\nSample spatial instructions:")
    for i in range(min(15, len(ds))):
        instr = ds[i].get("edit") or ds[i].get("instruction") or "N/A"
        print(f"  {i+1}. {instr[:120]}")

    # Check if images exist
    print("\nChecking image data...")
    sample = ds[0]
    if "input_image" in sample:
        img = sample["input_image"]
        print(f"  input_image: {type(img)}, size: {img.size if hasattr(img, 'size') else 'N/A'}")
    if "output_image" in sample:
        img = sample["output_image"]
        print(f"  output_image: {type(img)}, size: {img.size if hasattr(img, 'size') else 'N/A'}")

    print("\n✓ Dataset looks good!")

if __name__ == "__main__":
    main()
