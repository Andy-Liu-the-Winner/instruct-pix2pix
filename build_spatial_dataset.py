# build_spatial_dataset.py
#
# Uses LOCAL CACHE for HQ-Edit (no download)
# Streams AURORA from HuggingFace
# Filters for spatial instructions, combines, and saves

from datasets import load_dataset, Dataset, concatenate_datasets
import glob
import os

# STRICT spatial phrases, must indicate actual positioning/movement
SPATIAL_KWS = [
    # Positional phrases (require preposition + direction)
    "to the left", "to the right", "on the left", "on the right",
    "to the top", "to the bottom", "at the top", "at the bottom",
    "in the corner", "top left corner", "top right corner",
    "bottom left corner", "bottom right corner",

    # Relative positioning (actual spatial relationships)
    "in front of", "behind the", "next to the", "beside the",
    "on top of", "on the top of", "under the", "below the", "above the",
    "between the", "inside the", "outside the",

    # Movement with direction
    "move to", "move it to", "shift to", "drag to",
    "move left", "move right", "move up", "move down",
    "move closer", "move farther", "move away",

    # Placement with position
    "place on", "place in", "place behind", "place next to",
    "put on", "put in", "put behind", "put next to", "put in front",
    "position at", "position in",

    # Depth/layering
    "in the foreground", "in the background", "to the foreground", "to the background",
    "bring forward", "send backward", "in front", "in back",

    # Specific spatial actions
    "swap places", "switch positions", "rearrange",
    "closer to the camera", "farther from",
]


def is_spatial_instruction(text: str) -> bool:
    if text is None:
        return False
    text = text.lower()
    return any(k in text for k in SPATIAL_KWS)

def load_hq_edit_from_cache_streaming(out_dir: str, batch_size: int = 500):
    """Load HQ-Edit from local parquet cache - STREAMING to save memory."""
    parquet_pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--UCSC-VLAA--HQ-Edit/snapshots/*/data/*.parquet"
    )
    parquet_files = sorted(glob.glob(parquet_pattern))

    if not parquet_files:
        raise RuntimeError(f"No cached parquet files found at {parquet_pattern}")

    print(f"Found {len(parquet_files)} cached HQ-Edit parquet files")
    print(f"Processing files one by one to save memory...")

    os.makedirs(out_dir, exist_ok=True)

    total_spatial = 0
    shard_idx = 0
    batch = []

    for file_idx, pq_file in enumerate(parquet_files):
        print(f"Processing file {file_idx+1}/{len(parquet_files)}: {os.path.basename(pq_file)}")

        # Load one parquet file at a time
        ds = load_dataset("parquet", data_files=[pq_file], split="train")

        for ex in ds:
            instr = ex.get("edit") or ex.get("instruction") or ""
            if is_spatial_instruction(instr):
                batch.append(ex)

                # Save batch when full
                if len(batch) >= batch_size:
                    shard_ds = Dataset.from_list(batch)
                    shard_path = os.path.join(out_dir, f"shard_{shard_idx:04d}")
                    shard_ds.save_to_disk(shard_path)
                    total_spatial += len(batch)
                    print(f"  Saved shard {shard_idx} with {len(batch)} examples (total: {total_spatial})")
                    shard_idx += 1
                    batch = []
                    del shard_ds

        del ds  # Free memory

    # Save remaining batch
    if batch:
        shard_ds = Dataset.from_list(batch)
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:04d}")
        shard_ds.save_to_disk(shard_path)
        total_spatial += len(batch)
        print(f"  Saved final shard {shard_idx} with {len(batch)} examples")

    print(f"\nTotal spatial examples saved: {total_spatial}")
    return total_spatial

def stream_aurora_spatial(max_examples: int = 20000):
    """Stream AURORA and filter spatial instructions."""
    print("Streaming AURORA dataset...")

    try:
        stream = load_dataset("McGill-NLP/AURORA", split="train", streaming=True)
    except Exception as e:
        print(f"Warning: Could not load AURORA: {e}")
        return None

    buf = []
    for ex in stream:
        # Try different possible instruction keys
        instr = ex.get("instruction") or ex.get("edit") or ex.get("prompt") or ""
        if is_spatial_instruction(instr):
            buf.append(ex)
            if len(buf) >= max_examples:
                break
        if len(buf) % 1000 == 0 and len(buf) > 0:
            print(f"  Collected {len(buf)} spatial examples from AURORA...")

    print(f"Found {len(buf)} spatial examples from AURORA")
    return Dataset.from_list(buf) if buf else None

def main():
    # Stream and save - LOW MEMORY usage
    print("\n=== Loading HQ-Edit (streaming mode - low memory) ===")
    out_dir = "spatial_edits_hq_only"

    total = load_hq_edit_from_cache_streaming(out_dir, batch_size=500)

    print(f"\n{'='*50}")
    print(f"Saved spatial dataset to: {out_dir}")
    print(f"Total spatial examples: {total}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
