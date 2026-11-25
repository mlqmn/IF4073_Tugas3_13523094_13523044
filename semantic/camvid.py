import random
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CAMVID_DIR = ROOT / "data" / "camvid"
UCL_IMAGES_ZIP = CAMVID_DIR / "701_StillsRaw_full.zip"
UCL_LABELS_ZIP = CAMVID_DIR / "LabeledApproved_full.zip"
FASTAI_ARCHIVE = CAMVID_DIR / "camvid.tgz"

SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_SEED = 0

INPUT_SIZE = (360, 480)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = 255

ORIGINAL_CLASS_COLORS = {
    "Animal": (64, 128, 64), "Archway": (192, 0, 128), "Bicyclist": (0, 128, 192),
    "Bridge": (0, 128, 64), "Building": (128, 0, 0), "Car": (64, 0, 128),
    "CartLuggagePram": (64, 0, 192), "Child": (192, 128, 64), "Column_Pole": (192, 192, 128),
    "Fence": (64, 64, 128), "LaneMkgsDriv": (128, 0, 192), "LaneMkgsNonDriv": (192, 0, 64),
    "Misc_Text": (128, 128, 64), "MotorcycleScooter": (192, 0, 192), "OtherMoving": (128, 64, 64),
    "ParkingBlock": (64, 192, 128), "Pedestrian": (64, 64, 0), "Road": (128, 64, 128),
    "RoadShoulder": (128, 128, 192), "Sidewalk": (0, 0, 192), "SignSymbol": (192, 128, 128),
    "Sky": (128, 128, 128), "SUVPickupTruck": (64, 128, 192), "TrafficCone": (0, 0, 64),
    "TrafficLight": (0, 64, 64), "Train": (192, 64, 128), "Tree": (128, 128, 0),
    "Truck_Bus": (192, 128, 192), "Tunnel": (64, 0, 64), "VegetationMisc": (192, 192, 0),
    "Void": (0, 0, 0), "Wall": (64, 192, 0),
}

CLASS_GROUPS = {
    "Sky": ["Sky"],
    "Building": ["Bridge", "Building", "Wall", "Tunnel", "Archway"],
    "Pole": ["Column_Pole", "TrafficCone"],
    "Road": ["Road", "LaneMkgsDriv", "LaneMkgsNonDriv"],
    "Pavement": ["Sidewalk", "ParkingBlock", "RoadShoulder"],
    "Tree": ["Tree", "VegetationMisc"],
    "SignSymbol": ["SignSymbol", "Misc_Text", "TrafficLight"],
    "Fence": ["Fence"],
    "Car": ["Car", "SUVPickupTruck", "Truck_Bus", "Train", "OtherMoving"],
    "Pedestrian": ["Pedestrian", "Child", "CartLuggagePram", "Animal"],
    "Bicyclist": ["Bicyclist", "MotorcycleScooter"],
}
CLASS_NAMES = list(CLASS_GROUPS)
GROUP_INDEX_BY_ORIGINAL = {
    original: index for index, name in enumerate(CLASS_NAMES) for original in CLASS_GROUPS[name]
}


def rgb_key(rgb):
    rgb = np.asarray(rgb, dtype=np.int32)
    return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]


def build_rgb_lookup():
    lookup = np.full(1 << 24, IGNORE_INDEX, dtype=np.uint8)
    for name, color in ORIGINAL_CLASS_COLORS.items():
        lookup[rgb_key(color)] = GROUP_INDEX_BY_ORIGINAL.get(name, IGNORE_INDEX)
    return lookup


def build_code_lookup(codes_path):
    codes = codes_path.read_text().split()
    unknown = set(codes) - ORIGINAL_CLASS_COLORS.keys()
    if unknown:
        sys.exit(f"Nama kelas tidak dikenal di {codes_path}: {sorted(unknown)}")
    return np.array([GROUP_INDEX_BY_ORIGINAL.get(name, IGNORE_INDEX) for name in codes], dtype=np.uint8)


def extract_archives(raw_dir):
    if UCL_IMAGES_ZIP.exists() and UCL_LABELS_ZIP.exists():
        for archive in (UCL_IMAGES_ZIP, UCL_LABELS_ZIP):
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(raw_dir)
    elif FASTAI_ARCHIVE.exists():
        with tarfile.open(FASTAI_ARCHIVE) as tf:
            tf.extractall(raw_dir, filter="data")
    else:
        sys.exit(f"Letakkan {UCL_IMAGES_ZIP.name} + {UCL_LABELS_ZIP.name} atau {FASTAI_ARCHIVE.name} di {CAMVID_DIR}")


def collect_labeled_frames(raw_dir):
    images, labels = {}, {}
    for path in raw_dir.rglob("*.png"):
        stem, _, suffix = path.stem.rpartition("_")
        if suffix in ("L", "P"):
            labels[stem] = path
        else:
            images[path.stem] = path
    return {stem: (images[stem], labels[stem]) for stem in sorted(images.keys() & labels.keys())}


def read_grouped_label(label_path, rgb_lookup, code_lookup):
    if label_path.stem.endswith("_L"):
        return rgb_lookup[rgb_key(np.array(Image.open(label_path).convert("RGB")))]
    return code_lookup[np.array(Image.open(label_path))]


def assign_splits(stems):
    shuffled = sorted(stems)
    random.Random(SPLIT_SEED).shuffle(shuffled)
    train_end = round(len(shuffled) * SPLIT_RATIOS["train"])
    val_end = train_end + round(len(shuffled) * SPLIT_RATIOS["val"])
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def prepare_dataset():
    raw_dir = CAMVID_DIR / "raw"
    extract_archives(raw_dir)
    frames = collect_labeled_frames(raw_dir)
    rgb_lookup = build_rgb_lookup()
    codes_paths = list(raw_dir.rglob("codes.txt"))
    code_lookup = build_code_lookup(codes_paths[0]) if codes_paths else None

    for split, stems in assign_splits(frames).items():
        images_dir = CAMVID_DIR / split / "images"
        labels_dir = CAMVID_DIR / split / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            image_path, label_path = frames[stem]
            grouped_label = read_grouped_label(label_path, rgb_lookup, code_lookup)
            Image.fromarray(grouped_label).save(labels_dir / f"{stem}.png")
            shutil.move(image_path, images_dir / f"{stem}.png")
        print(f"{split}: {len(stems)} citra")

    shutil.rmtree(raw_dir)


def normalize_image(image):
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255).permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (pixels - mean) / std


class CamVidSplit(torch.utils.data.Dataset):
    def __init__(self, split, random_flip=False):
        self.image_paths = sorted((CAMVID_DIR / split / "images").glob("*.png"))
        self.labels_dir = CAMVID_DIR / split / "labels"
        self.random_flip = random_flip
        if not self.image_paths:
            raise FileNotFoundError(f"Split '{split}' kosong. Jalankan camvid.py terlebih dahulu.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        height, width = INPUT_SIZE
        image = Image.open(image_path).convert("RGB").resize((width, height), Image.BILINEAR)
        label = Image.open(self.labels_dir / image_path.name).resize((width, height), Image.NEAREST)
        if self.random_flip and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return normalize_image(image), torch.from_numpy(np.array(label, dtype=np.int64))


if __name__ == "__main__":
    prepare_dataset()
