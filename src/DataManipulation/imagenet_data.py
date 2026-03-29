"""
ImageNet-1K (ILSVRC2012) download and preparation utility.

Downloads ImageNet-1K from HuggingFace and converts to ImageFolder format:
    data_dir/train/{class_wnid}/images...
    data_dir/val/{class_wnid}/images...

Requires HuggingFace authentication:
    huggingface-cli login
    # Then accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k

Usage:
    python src/DataManipulation/imagenet_data.py --output_dir ./data/ImageNet
"""

import os
import argparse
from pathlib import Path
from PIL import Image
from datasets import load_dataset


def download_and_prepare_imagenet(output_dir, num_proc=8):
    """
    Download ImageNet-1K from HuggingFace and save in ImageFolder format.

    Arguments:
    ----------
    output_dir: str
        Root directory where the dataset will be stored.
    num_proc: int
        Number of parallel processes for download/processing.

    Returns:
    --------
    train_dir: str
        Path to the training directory.
    val_dir: str
        Path to the validation directory.
    """
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    marker_file = output_dir / ".prepared"

    # If already prepared, return paths directly
    if marker_file.exists():
        print("ImageNet-1K already downloaded and prepared.")
        n_train = sum(1 for _ in train_dir.rglob("*.JPEG")) if train_dir.exists() else 0
        n_val = sum(1 for _ in val_dir.rglob("*.JPEG")) if val_dir.exists() else 0
        print(f"  Train: {n_train} images, Val: {n_val} images")
        return str(train_dir), str(val_dir)

    print("Downloading ImageNet-1K from HuggingFace...")
    print("This requires authentication. Run 'huggingface-cli login' first.")
    print("Also accept terms at: https://huggingface.co/datasets/ILSVRC/imagenet-1k")

    # Download both splits
    for split_name, split_dir in [("train", train_dir), ("validation", val_dir)]:
        print(f"\nProcessing {split_name} split...")
        split_dir.mkdir(parents=True, exist_ok=True)

        # Load the dataset split (streams from HF Hub)
        # Note: num_proc>1 can deadlock on large datasets, use single process
        ds = load_dataset(
            "ILSVRC/imagenet-1k",
            split=split_name,
        )

        # Get class label mapping
        label_names = ds.features["label"].names  # List of wnids/class names

        # Create class directories
        for label_name in label_names:
            (split_dir / label_name).mkdir(exist_ok=True)

        # Save images to class directories
        total = len(ds)
        print(f"Saving {total} images to {split_dir}...")
        import time
        t0 = time.time()
        for idx, example in enumerate(ds):
            img = example["image"]
            label_idx = example["label"]
            class_name = label_names[label_idx]

            # Convert to RGB if needed
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Save as JPEG
            img_path = split_dir / class_name / f"{split_name}_{idx:08d}.JPEG"
            img.save(img_path, "JPEG", quality=95)

            if (idx + 1) % 5000 == 0:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (total - idx - 1) / rate
                print(f"  [{idx+1}/{total}] {rate:.0f} img/s, ETA: {eta/60:.0f}min", flush=True)

        print(f"  {split_name} complete: {total} images saved")

    # Write marker file
    marker_file.touch()
    print("\nImageNet-1K preparation complete.")

    return str(train_dir), str(val_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare ImageNet-1K")
    parser.add_argument("--output_dir", type=str, default="./data/ImageNet",
                        help="Output directory for ImageNet data")
    parser.add_argument("--num_proc", type=int, default=8,
                        help="Number of parallel download processes")
    args = parser.parse_args()

    train_dir, val_dir = download_and_prepare_imagenet(args.output_dir, args.num_proc)
    print(f"\nTrain dir: {train_dir}")
    print(f"Val dir: {val_dir}")
