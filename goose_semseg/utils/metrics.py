from __future__ import annotations

from typing import Dict, List, Union

import torch


def update_confusion_matrix(
    confusion_matrix: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    predictions = predictions.detach().reshape(-1).to(torch.int64)
    targets = targets.detach().reshape(-1).to(torch.int64)
    valid = (targets != ignore_index) & (targets >= 0) & (targets < num_classes)
    if not torch.any(valid):
        return
    indices = targets[valid] * num_classes + predictions[valid].clamp(0, num_classes - 1)
    confusion_matrix += torch.bincount(indices, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def compute_mean_iou(confusion_matrix: torch.Tensor) -> float:
    matrix = confusion_matrix.float()
    intersection = torch.diag(matrix)
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - intersection
    valid = union > 0
    if not torch.any(valid):
        return 0.0
    return float((intersection[valid] / union[valid].clamp_min(1)).mean().item())


def _safe_metric(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def per_class_metric_rows(confusion_matrix: torch.Tensor, epoch: int) -> List[Dict[str, Union[int, float, bool, str]]]:
    matrix = confusion_matrix.detach().cpu().to(torch.float64)
    true_positive = torch.diag(matrix)
    gt_pixels = matrix.sum(dim=1)
    pred_pixels = matrix.sum(dim=0)
    false_positive = pred_pixels - true_positive
    false_negative = gt_pixels - true_positive
    union = gt_pixels + pred_pixels - true_positive
    total_valid_pixels = float(matrix.sum().item())

    rows: List[Dict[str, Union[int, float, bool, str]]] = []
    for class_id in range(matrix.shape[0]):
        tp = float(true_positive[class_id].item())
        fp = float(false_positive[class_id].item())
        fn = float(false_negative[class_id].item())
        gt_count = float(gt_pixels[class_id].item())
        pred_count = float(pred_pixels[class_id].item())
        union_count = float(union[class_id].item())
        rows.append(
            {
                "epoch": epoch + 1,
                "class_id": class_id,
                "class_name": f"class_{class_id}",
                "gt_pixels": int(gt_count),
                "pred_pixels": int(pred_count),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "precision": _safe_metric(tp, pred_count),
                "recall": _safe_metric(tp, gt_count),
                "iou": _safe_metric(tp, union_count),
                "gt_pixel_ratio": _safe_metric(gt_count, total_valid_pixels),
                "pred_pixel_ratio": _safe_metric(pred_count, total_valid_pixels),
                "gt_present": gt_count > 0,
                "pred_present": pred_count > 0,
                "gt_present_pred_missing": gt_count > 0 and pred_count == 0,
                "pred_present_gt_missing": pred_count > 0 and gt_count == 0,
            }
        )
    return rows
