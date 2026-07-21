from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from goose_semseg.losses.m2f_criterion import Mask2FormerSetCriterion
from goose_semseg.utils.metrics import per_class_metric_rows


LOSS_FIELDS = (
    "loss_ce",
    "loss_mask",
    "loss_dice",
    "aux_loss_ce",
    "aux_loss_mask",
    "aux_loss_dice",
    "loss_cls_aux",
)
CLS_METRIC_FIELDS = ("cls_aux_accuracy",)


def append_per_class_metrics(path: Path, epoch: int, confusion_matrix: torch.Tensor) -> None:
    rows = per_class_metric_rows(confusion_matrix, epoch)
    fieldnames = [
        "epoch",
        "class_id",
        "class_name",
        "gt_pixels",
        "pred_pixels",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "iou",
        "gt_pixel_ratio",
        "pred_pixel_ratio",
        "gt_present",
        "pred_present",
        "gt_present_pred_missing",
        "pred_present_gt_missing",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_presence_summary(path: Path, epoch: int, confusion_matrix: torch.Tensor) -> None:
    rows = per_class_metric_rows(confusion_matrix, epoch)
    gt_present_pred_missing = [
        str(row["class_id"]) for row in rows if bool(row["gt_present_pred_missing"])
    ]
    pred_present_gt_missing = [
        str(row["class_id"]) for row in rows if bool(row["pred_present_gt_missing"])
    ]
    low_recall_classes = [
        str(row["class_id"])
        for row in rows
        if bool(row["gt_present"])
        and isinstance(row["recall"], float)
        and not np.isnan(row["recall"])
        and row["recall"] < 0.1
    ]
    low_precision_classes = [
        str(row["class_id"])
        for row in rows
        if bool(row["pred_present"])
        and isinstance(row["precision"], float)
        and not np.isnan(row["precision"])
        and row["precision"] < 0.1
    ]
    summary = {
        "epoch": epoch + 1,
        "gt_present_pred_missing_count": len(gt_present_pred_missing),
        "gt_present_pred_missing_class_ids": "|".join(gt_present_pred_missing),
        "pred_present_gt_missing_count": len(pred_present_gt_missing),
        "pred_present_gt_missing_class_ids": "|".join(pred_present_gt_missing),
        "low_recall_lt_0_1_count": len(low_recall_classes),
        "low_recall_lt_0_1_class_ids": "|".join(low_recall_classes),
        "low_precision_lt_0_1_count": len(low_precision_classes),
        "low_precision_lt_0_1_class_ids": "|".join(low_precision_classes),
    }
    fieldnames = list(summary.keys())
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


def zero_breakdown() -> Dict[str, float]:
    return {field: 0.0 for field in LOSS_FIELDS}


def zero_cls_metrics() -> Dict[str, float]:
    return {field: float("nan") for field in CLS_METRIC_FIELDS}


def update_loss_breakdown(
    total: Dict[str, float],
    loss_dict: Dict[str, torch.Tensor],
    criterion: Mask2FormerSetCriterion,
) -> None:
    for key, value in loss_dict.items():
        if value is None:
            continue
        scalar = float(value.detach().item())
        if key == "loss_cls_aux":
            total["loss_cls_aux"] += scalar
        elif key.startswith("loss_ce"):
            scalar *= float(criterion.class_weight)
            total["loss_ce" if key == "loss_ce" else "aux_loss_ce"] += scalar
        elif key.startswith("loss_mask"):
            scalar *= float(criterion.mask_weight)
            total["loss_mask" if key == "loss_mask" else "aux_loss_mask"] += scalar
        elif key.startswith("loss_dice"):
            scalar *= float(criterion.dice_weight)
            total["loss_dice" if key == "loss_dice" else "aux_loss_dice"] += scalar


def append_epoch_log(
    path: Path,
    epoch: int,
    train_loss: float,
    train_miou: float,
    val_loss: float,
    val_miou: float,
    best_val_miou: float,
    train_breakdown: Dict[str, float],
    val_breakdown: Dict[str, float],
    train_cls_metrics: Dict[str, float],
    val_cls_metrics: Dict[str, float],
) -> None:
    exists = path.exists()
    fieldnames = [
        "epoch",
        "train_loss",
        "train_miou",
        "val_loss",
        "val_miou",
        "best_val_miou",
    ]
    fieldnames.extend(f"train_{field}" for field in LOSS_FIELDS)
    fieldnames.extend(f"val_{field}" for field in LOSS_FIELDS)
    fieldnames.extend(f"train_{field}" for field in CLS_METRIC_FIELDS)
    fieldnames.extend(f"val_{field}" for field in CLS_METRIC_FIELDS)
    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_miou": train_miou,
        "val_loss": val_loss,
        "val_miou": val_miou,
        "best_val_miou": best_val_miou,
    }
    row.update({f"train_{field}": train_breakdown[field] for field in LOSS_FIELDS})
    row.update({f"val_{field}": val_breakdown[field] for field in LOSS_FIELDS})
    row.update({f"train_{field}": train_cls_metrics[field] for field in CLS_METRIC_FIELDS})
    row.update({f"val_{field}": val_cls_metrics[field] for field in CLS_METRIC_FIELDS})
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
