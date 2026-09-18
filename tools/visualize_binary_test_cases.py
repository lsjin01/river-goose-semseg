#!/usr/bin/env python3
"""Rank binary test frames and render Image/GT/Predict/Error panels without TTA."""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party")]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch
from PIL import Image
from torch.utils.data import DataLoader

from goose_semseg.data.binary_algae import BinaryAlgaeTileDataset
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import BackboneLayersSet, build_segmentation_decoder
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.pretrained.mask2former import (
    infer_pretrained_feature_channels,
    load_pretrained_mask2former,
)
from goose_semseg.utils.seed import seed_everything


CLASS_NAMES = ("non_algae", "algae_including_nps_algae")
CLASS_COLORS = np.asarray(((105, 120, 135), (35, 190, 80)), dtype=np.uint8)
FP_COLOR = np.asarray((255, 165, 0), dtype=np.uint8)
FN_COLOR = np.asarray((235, 40, 55), dtype=np.uint8)


def encode_mask(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(mask).save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def decode_mask(payload: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(payload))).copy()


def confusion_and_iou(gt: np.ndarray, pred: np.ndarray):
    valid = gt != 255
    encoded = gt[valid].astype(np.int64) * 2 + pred[valid].astype(np.int64)
    cm = np.bincount(encoded, minlength=4).reshape(2, 2)
    intersection = np.diag(cm).astype(np.float64)
    union = cm.sum(0) + cm.sum(1) - intersection
    iou = np.divide(intersection, union, out=np.full(2, np.nan), where=union > 0)
    return cm, iou, float(np.nanmean(iou))


def altum_preview(path: Path) -> Image.Image:
    channels = []
    with Image.open(path) as image:
        for page in (2, 1, 0):
            image.seek(page)
            values = np.asarray(image, dtype=np.float32)
            low, high = np.percentile(values, (1, 99))
            scaled = np.clip((values - low) / max(float(high - low), 1.0), 0, 1)
            channels.append(np.uint8(scaled * 255))
    return Image.fromarray(np.stack(channels, axis=-1))


def source_preview(dataset, record) -> tuple[Image.Image, str]:
    path = dataset.source / record["source_image"]
    if record["sensor"] == "P1":
        return Image.open(path).convert("RGB"), "Image (P1 RGB)"
    return altum_preview(path), "Image (Altum display composite: page2/1/0 → R/G/B)"


def colorize(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    output = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        output[valid & (mask == class_id)] = color
    return output


def resize_array(array: np.ndarray, size: tuple[int, int], nearest: bool) -> np.ndarray:
    resampling = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    return np.asarray(Image.fromarray(array).resize(size, resampling))


def render_case(dataset, case: dict, output: Path, label: str, rank: int) -> str:
    record = dataset.records[case["frame_index"]]
    gt, pred = decode_mask(case["gt_png"]), decode_mask(case["pred_png"])
    valid = gt != 255
    source, source_title = source_preview(dataset, record)

    max_width, max_height = 1200, 850
    scale = min(max_width / gt.shape[1], max_height / gt.shape[0], 1.0)
    display_size = (max(1, round(gt.shape[1] * scale)), max(1, round(gt.shape[0] * scale)))
    source = np.asarray(source.resize(display_size, Image.Resampling.LANCZOS))
    gt_display = resize_array(gt, display_size, nearest=True)
    pred_display = resize_array(pred, display_size, nearest=True)
    valid_display = gt_display != 255

    gt_color = colorize(gt_display, valid_display)
    pred_color = colorize(pred_display, valid_display)
    error = np.uint8(source.astype(np.float32) * 0.28)
    fp = valid_display & (gt_display == 0) & (pred_display == 1)
    fn = valid_display & (gt_display == 1) & (pred_display == 0)
    error[fp] = FP_COLOR
    error[fn] = FN_COLOR
    error[~valid_display] = 0

    fig, axes = plt.subplots(1, 4, figsize=(24, 7.2))
    panels = (
        (source, source_title),
        (gt_color, "GT"),
        (pred_color, "Predict"),
        (error, "Error (orange: false algae, red: missed algae)"),
    )
    for axis, (image, title) in zip(axes, panels):
        axis.imshow(image)
        axis.set_title(title, fontsize=11)
        axis.axis("off")

    relative_source = record["source_image"]
    fig.suptitle(
        f"{label.title()} #{rank} | {relative_source}\n"
        f"frame mIoU={case['miou']:.4f} | non-algae IoU={case['iou'][0]:.4f} | "
        f"algae IoU={case['iou'][1]:.4f} | no TTA",
        fontsize=14,
    )
    legend = [
        Patch(color=CLASS_COLORS[0] / 255, label=CLASS_NAMES[0]),
        Patch(color=CLASS_COLORS[1] / 255, label=CLASS_NAMES[1]),
        Patch(color=FP_COLOR / 255, label="false algae (GT non-algae → predicted algae)"),
        Patch(color=FN_COLOR / 255, label="missed algae (GT algae → predicted non-algae)"),
        Patch(color=(0, 0, 0), label="ignored / padded"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.12, wspace=0.025)
    filename = f"{label}_{rank}_frame_{case['frame_index']:04d}.png"
    fig.savefig(output / filename, dpi=150)
    plt.close(fig)
    return filename


def build_model(training, checkpoint: dict, device: torch.device):
    backbone = load_dinov3_backbone(training)
    pretrained = load_pretrained_mask2former(training.mask2former_pretrained_model_name_or_path)
    model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type="m2f",
        hidden_dim=training.hidden_dim,
        num_classes=training.num_classes,
        autocast_dtype=torch.bfloat16,
        freeze_backbone=training.freeze_backbone,
        feature_channels=infer_pretrained_feature_channels(pretrained),
        input_channels=training.input_channels,
        cls_aux_num_classes=(training.cls_aux_num_classes if training.enable_cls_aux else 0),
        fusion_type=training.fusion_type,
        precomputed_indices=True,
    )
    del pretrained
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output_dir).resolve()
    if (output / "report.json").exists():
        raise FileExistsError(f"Completed report already exists: {output / 'report.json'}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    training = argparse.Namespace(**checkpoint["args"])
    if training.num_classes != 2 or training.segmentation_taxonomy != "binary_algae":
        raise ValueError("Expected a two-class binary_algae checkpoint")
    seed_everything(training.seed)
    dataset = BinaryAlgaeTileDataset(ROOT / training.data_path, "test", training.tile_size)
    model = build_model(training, checkpoint, device)
    del checkpoint
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=1,
    )

    best: list[dict] = []
    worst: list[dict] = []
    scores: list[dict] = []
    current_index = None
    current_gt = current_pred = None
    offset = 0

    def finalize(frame_index: int, gt: np.ndarray, pred: np.ndarray) -> None:
        nonlocal best, worst
        cm, iou, miou = confusion_and_iou(gt, pred)
        case = {
            "frame_index": frame_index,
            "sensor": dataset.records[frame_index]["sensor"],
            "source_image": dataset.records[frame_index]["source_image"],
            "miou": miou,
            "iou": iou.tolist(),
            "confusion_matrix": cm.tolist(),
        }
        scores.append(case)
        candidate = dict(case, gt_png=encode_mask(gt), pred_png=encode_mask(pred))
        best = sorted(best + [candidate], key=lambda item: item["miou"], reverse=True)[:2]
        worst = sorted(worst + [candidate], key=lambda item: item["miou"])[:2]

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = mask2former_semantic_scores(
                    model(images), target_size=targets.shape[-2:]
                ).argmax(1).cpu()
            for j in range(len(targets)):
                frame_index, left, top = dataset.tiles[offset + j]
                record = dataset.records[frame_index]
                if current_index != frame_index:
                    if current_index is not None:
                        finalize(current_index, current_gt, current_pred)
                    current_index = frame_index
                    current_gt = np.full((record["height"], record["width"]), 255, dtype=np.uint8)
                    current_pred = np.zeros((record["height"], record["width"]), dtype=np.uint8)
                width = min(dataset.tile_size, record["width"] - left)
                height = min(dataset.tile_size, record["height"] - top)
                current_gt[top : top + height, left : left + width] = targets[j, :height, :width].numpy()
                current_pred[top : top + height, left : left + width] = predictions[
                    j, :height, :width
                ].numpy()
            offset += len(targets)
            if offset % 500 < len(targets) or offset == len(dataset):
                print(f"ranked tiles {offset}/{len(dataset)}", flush=True)
    if current_index is not None:
        finalize(current_index, current_gt, current_pred)
    if offset != len(dataset) or len(scores) != len(dataset.records):
        raise RuntimeError("Incomplete test traversal")

    best_ids = {case["frame_index"] for case in best}
    worst = [case for case in worst if case["frame_index"] not in best_ids]
    if len(worst) != 2:
        raise RuntimeError("Best and worst selections overlap")
    files = []
    for label, cases in (("best", best), ("worst", worst)):
        for rank, case in enumerate(cases, 1):
            files.append(render_case(dataset, case, output, label, rank))

    with (output / "frame_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("frame_index", "sensor", "source_image", "miou", "non_algae_iou", "algae_iou"),
        )
        writer.writeheader()
        for row in sorted(scores, key=lambda item: item["miou"], reverse=True):
            writer.writerow(
                dict(
                    frame_index=row["frame_index"],
                    sensor=row["sensor"],
                    source_image=row["source_image"],
                    miou=row["miou"],
                    non_algae_iou=row["iou"][0],
                    algae_iou=row["iou"][1],
                )
            )
    report = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "split": "test",
        "tta": False,
        "ranking_metric": "full-frame binary mIoU over GT-present classes",
        "frames": len(scores),
        "tiles": offset,
        "best": [{k: v for k, v in case.items() if not k.endswith("_png")} for case in best],
        "worst": [{k: v for k, v in case.items() if not k.endswith("_png")} for case in worst],
        "visualizations": files,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
