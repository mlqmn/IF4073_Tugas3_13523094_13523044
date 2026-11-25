import json

import torch

from camvid import CamVidSplit
from model import WEIGHTS_PATH, confusion_matrix, load_trained_model, segmentation_scores

METRICS_PATH = WEIGHTS_PATH.with_name("test_metrics.json")
BATCH_SIZE = 8


def evaluate_test_split(device="cpu"):
    model = load_trained_model(device)
    loader = torch.utils.data.DataLoader(CamVidSplit("test"), BATCH_SIZE)
    scores = segmentation_scores(confusion_matrix(model, loader, device))
    METRICS_PATH.write_text(json.dumps(scores, indent=2))

    print(f"global accuracy {scores['global_accuracy']:.4f}")
    print(f"mean accuracy   {scores['mean_accuracy']:.4f}")
    print(f"mean IoU        {scores['mean_iou']:.4f}")
    for name, iou in scores["per_class_iou"].items():
        print(f"  {name:<11} {iou:.4f}")
    return scores


if __name__ == "__main__":
    evaluate_test_split("cuda" if torch.cuda.is_available() else "cpu")
