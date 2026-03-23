import os
import zipfile
import shutil
import urllib.request
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Dataset


TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_and_prepare_tinyimagenet(root_dir):
    """
    Download Tiny-ImageNet-200 and reorganize into ImageFolder-compatible structure.

    After preparation, the directory structure is:
        root_dir/tiny-imagenet-200/train/{class_id}/*.JPEG
        root_dir/tiny-imagenet-200/val/{class_id}/*.JPEG

    Arguments:
    ----------
    root_dir: str
        Root directory where the dataset will be stored.

    Returns:
    --------
    train_dir: str
        Path to the reorganized training directory.
    val_dir: str
        Path to the reorganized validation directory (used as test set).
    """
    root_dir = Path(root_dir)
    dataset_dir = root_dir / "tiny-imagenet-200"
    zip_path = root_dir / "tiny-imagenet-200.zip"
    marker_file = dataset_dir / ".prepared"

    # If already prepared, return paths directly
    if marker_file.exists():
        print("Tiny-ImageNet already downloaded and prepared.")
        return str(dataset_dir / "train"), str(dataset_dir / "val")

    # Download if zip doesn't exist
    if not zip_path.exists():
        print(f"Downloading Tiny-ImageNet-200 to {zip_path}...")
        root_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TINYIMAGENET_URL, str(zip_path))
        print("Download complete.")

    # Extract if dataset directory doesn't exist
    if not dataset_dir.exists():
        print(f"Extracting Tiny-ImageNet-200...")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(root_dir))
        print("Extraction complete.")

    # Reorganize train: move images from {class}/images/*.JPEG to {class}/*.JPEG
    train_dir = dataset_dir / "train"
    print("Reorganizing training data...")
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        images_dir = class_dir / "images"
        if images_dir.exists():
            for img_file in images_dir.iterdir():
                img_file.rename(class_dir / img_file.name)
            images_dir.rmdir()
        # Remove annotation boxes file if present
        boxes_file = class_dir / (class_dir.name + "_boxes.txt")
        if boxes_file.exists():
            boxes_file.unlink()

    # Reorganize val: flat images/ dir -> class-based subdirectories
    val_dir = dataset_dir / "val"
    val_images_dir = val_dir / "images"
    val_annotations = val_dir / "val_annotations.txt"

    if val_images_dir.exists() and val_annotations.exists():
        print("Reorganizing validation data...")
        # Parse annotations: filename -> class_id
        with open(str(val_annotations), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                filename = parts[0]
                class_id = parts[1]
                class_dir = val_dir / class_id
                class_dir.mkdir(exist_ok=True)
                src = val_images_dir / filename
                dst = class_dir / filename
                if src.exists():
                    src.rename(dst)
        # Clean up empty images dir and annotations
        if val_images_dir.exists():
            shutil.rmtree(str(val_images_dir), ignore_errors=True)
        val_annotations.unlink(missing_ok=True)

    # Write marker file
    marker_file.touch()
    print("Tiny-ImageNet preparation complete.")

    return str(train_dir), str(val_dir)


def put_TinyImageNet_data_generic_form(imagefolder_data):
    """
    Put the data of a torchvision ImageFolder dataset under the same format as
    the rest of the datasets.

    Arguments:
    ----------
    imagefolder_data: list of (tensor, label) tuples

    Returns:
    --------
    generic_data: dict
        Dict where the keys are the ids of the samples and the values are
        also dictionaries with two keys: 'Data' and 'Label'
    """
    generic_data = {}
    for id_current_sample in range(len(imagefolder_data)):
        generic_data[id_current_sample] = {
            'Data': imagefolder_data[id_current_sample][0],
            'Label': imagefolder_data[id_current_sample][1]
        }
    return generic_data


class TinyImageNetDatasetWrapper(Dataset):
    """
    Tiny-ImageNet dataset wrapper.

    Argument:
    ---------
    data: dict
        Dict where the keys are the ids of the samples and the values are
        also dictionaries with two keys: 'Data' and 'Label'
    """
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample_data, label = self.data[i]['Data'], self.data[i]['Label']
        return sample_data, label
