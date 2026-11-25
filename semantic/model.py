from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

from camvid import CLASS_NAMES, IGNORE_INDEX, INPUT_SIZE, normalize_image

WEIGHTS_PATH = Path(__file__).resolve().parent / "weights" / "deeplabv3_mobilenet_camvid.pt"


def build_model(pretrained=True):
    if pretrained:
        model = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1)
    else:
        model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None)
    model.aux_classifier = None
    head_in_channels = model.classifier[-1].in_channels
    model.classifier[-1] = torch.nn.Conv2d(head_in_channels, len(CLASS_NAMES), kernel_size=1)
    return model


def load_trained_model(device="cpu"):
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Bobot model belum ada: {WEIGHTS_PATH}. Jalankan train.py terlebih dahulu.")
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    return model.to(device).eval()


@torch.no_grad()
def predict_class_map(model, image, device="cpu"):
    original_width, original_height = image.size
    height, width = INPUT_SIZE
    batch = normalize_image(image.convert("RGB").resize((width, height), Image.BILINEAR))
    logits = model(batch.unsqueeze(0).to(device))["out"]
    logits = F.interpolate(logits, size=(original_height, original_width), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def confusion_matrix(model, loader, device="cpu"):
    num_classes = len(CLASS_NAMES)
    counts = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for images, labels in loader:
        predictions = model(images.to(device))["out"].argmax(dim=1).cpu()
        labeled = labels != IGNORE_INDEX
        pair_index = labels[labeled] * num_classes + predictions[labeled]
        counts += torch.bincount(pair_index, minlength=num_classes**2).view(num_classes, num_classes)
    return counts.numpy()


def segmentation_scores(confusion):
    true_positive = np.diag(confusion).astype(np.float64)
    union = confusion.sum(0) + confusion.sum(1) - true_positive
    present = confusion.sum(1) > 0
    per_class_iou = true_positive / np.maximum(union, 1)
    per_class_accuracy = true_positive / np.maximum(confusion.sum(1), 1)
    # Classes absent from the ground truth are left out of the means so they do not count as IoU 0.
    return {
        "global_accuracy": true_positive.sum() / max(confusion.sum(), 1),
        "mean_accuracy": per_class_accuracy[present].mean() if present.any() else 0.0,
        "mean_iou": per_class_iou[present].mean() if present.any() else 0.0,
        "per_class_iou": dict(zip(CLASS_NAMES, per_class_iou.round(4).tolist())),
    }
