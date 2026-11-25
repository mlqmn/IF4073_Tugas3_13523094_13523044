import numpy as np
import torch

from camvid import CLASS_NAMES, IGNORE_INDEX, CamVidSplit
from evaluate import evaluate_test_split
from model import WEIGHTS_PATH, build_model, confusion_matrix, segmentation_scores

EPOCHS = 30
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4


def median_frequency_weights(dataset):
    num_classes = len(CLASS_NAMES)
    class_pixels = np.zeros(num_classes)
    pixels_of_images_with_class = np.zeros(num_classes)
    for _, label in dataset:
        counts = np.bincount(label[label != IGNORE_INDEX].numpy(), minlength=num_classes)
        class_pixels += counts
        pixels_of_images_with_class += (counts > 0) * counts.sum()
    frequency = class_pixels / np.maximum(pixels_of_images_with_class, 1)
    # Median frequency balancing (Eigen & Fergus, 2015) keeps rare classes such as Pole and Bicyclist from vanishing.
    # Classes missing from the training split get weight 1 and are left out of the median.
    present = frequency > 0
    weights = np.ones(num_classes)
    if present.any():
        weights[present] = np.median(frequency[present]) / frequency[present]
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images)["out"], labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = CamVidSplit("train", random_flip=True)
    val_set = CamVidSplit("val")
    train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_set, BATCH_SIZE, num_workers=NUM_WORKERS)

    model = build_model(pretrained=True).to(device)
    class_weights = median_frequency_weights(CamVidSplit("train")).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    WEIGHTS_PATH.parent.mkdir(exist_ok=True)
    best_val_iou = -1.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        model.eval()
        val_iou = segmentation_scores(confusion_matrix(model, val_loader, device))["mean_iou"]
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"epoch {epoch:02d}  train loss {train_loss:.4f}  val mIoU {val_iou:.4f}  best {best_val_iou:.4f}")

    evaluate_test_split(device)


if __name__ == "__main__":
    main()
