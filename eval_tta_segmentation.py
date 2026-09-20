#!/usr/bin/env python3
"""Evaluate segmentation checkpoints with optional hflip/multi-scale TTA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from train_student_kd import (
    build_dataset,
    build_student,
    count_parameters,
    forward_student,
    load_teacher,
    mask2former_outputs_to_semantic_probs,
    model_size_mb,
)


COARSE_CLASS_NAMES = ["background", "농업계", "축산계", "하천수면/수변"]
FINE_CLASS_NAMES = [
    "background",
    "밭_논",
    "잔재물",
    "배수로",
    "비닐하우스",
    "과수원",
    "축사",
    "야적퇴비_가축분뇨",
    "목장",
    "분뇨개별처리시설",
    "부유쓰레기",
    "연못",
]


def class_names_for(num_classes: int) -> List[str]:
    if num_classes == len(COARSE_CLASS_NAMES):
        return COARSE_CLASS_NAMES
    if num_classes == len(FINE_CLASS_NAMES):
        return FINE_CLASS_NAMES
    return [f"class_{idx}" for idx in range(num_classes)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["student", "teacher"], required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--student_model", choices=["segformer_b0", "segformer_b1"], default="segformer_b1")
    parser.add_argument("--student_local_files_only", action="store_true")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--resize_width", type=int, default=512)
    parser.add_argument("--resize_height", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--hflip", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--output_json", default=None)

    # Dataset compatibility args used by build_dataset().
    parser.add_argument("--dataset_factory", default=None)
    parser.add_argument("--train_image_dir", default=None)
    parser.add_argument("--train_mask_dir", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_mask_dir", default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--flip_prob", type=float, default=0.0)
    parser.add_argument("--enable_random_crop", action="store_true")
    parser.add_argument("--crop_width", type=int, default=1024)
    parser.add_argument("--crop_height", type=int, default=1024)
    parser.add_argument("--enable_rare_class_crop", action="store_true")
    parser.add_argument("--rare_class_crop_prob", type=float, default=0.3)
    parser.add_argument("--rare_class_ids", type=int, nargs="*", default=[])
    parser.add_argument("--rare_class_min_pixels", type=int, default=0)
    parser.add_argument("--rare_class_min_ratio", type=float, default=0.005)
    parser.add_argument("--rare_class_crop_attempts", type=int, default=10)

    # Teacher reconstruction args.
    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--teacher_model_factory", default="goose")
    parser.add_argument("--teacher_strict_load", action="store_true")
    parser.add_argument("--backbone_loader", choices=["auto", "native", "hf"], default="auto")
    parser.add_argument(
        "--hf_dinov3_model_name_or_path",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument("--allow_hf_download", dest="hf_local_files_only", action="store_false")
    parser.set_defaults(hf_local_files_only=True)
    parser.add_argument("--dinov3_weights", default=None)
    parser.add_argument(
        "--mask2former_pretrained_model_name_or_path",
        default="facebook/mask2former-swin-large-ade-semantic",
    )
    parser.add_argument("--disable_mask2former_pretrained", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--enable_cls_aux", action="store_true")
    parser.add_argument("--cls_aux_num_classes", type=int, default=0)
    return parser.parse_args()


def load_student_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model_type == "teacher":
        args.teacher_checkpoint = args.checkpoint
        model = load_teacher(args, device)
    else:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        saved_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        model = build_student(
            args.student_model,
            args.num_classes,
            local_files_only=args.student_local_files_only,
            input_channels=int(saved_args.get("input_channels", 3)),
            precomputed_indices=bool(saved_args.get("precomputed_indices", False)),
        )
        state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
    return model


@torch.no_grad()
def forward_dense(
    model: nn.Module,
    model_type: str,
    images: torch.Tensor,
    num_classes: int,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    if model_type == "teacher":
        outputs = model(images)
        return mask2former_outputs_to_semantic_probs(
            outputs,
            num_classes=num_classes,
            target_size=target_size,
            temperature=1.0,
        )
    return forward_student(model, images, target_size)


@torch.no_grad()
def tta_forward(
    model: nn.Module,
    model_type: str,
    images: torch.Tensor,
    num_classes: int,
    target_size: Tuple[int, int],
    scales: Iterable[float],
    hflip: bool,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"Scale must be positive. Got: {scale}")
        if scale == 1.0:
            scaled = images
        else:
            scaled_size = (
                max(1, int(round(images.shape[-2] * scale))),
                max(1, int(round(images.shape[-1] * scale))),
            )
            scaled = F.interpolate(
                images,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
            )
        outputs.append(forward_dense(model, model_type, scaled, num_classes, target_size))
        if hflip:
            flipped = torch.flip(scaled, dims=[-1])
            flipped_output = forward_dense(model, model_type, flipped, num_classes, target_size)
            outputs.append(torch.flip(flipped_output, dims=[-1]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def update_confusion(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> Tuple[int, int]:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return 0, 0
    correct = int((preds[valid] == labels[valid]).sum().item())
    total = int(valid.sum().item())
    inds = num_classes * labels[valid].reshape(-1) + preds[valid].reshape(-1)
    confusion += torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)
    return correct, total


def metrics_from_confusion(confusion: torch.Tensor, correct: int, total: int) -> Dict[str, Any]:
    confusion_f = confusion.float()
    tp = confusion_f.diag()
    union = confusion_f.sum(0) + confusion_f.sum(1) - tp
    iou = tp / union.clamp_min(1.0)
    return {
        "mIoU": float(iou.mean().item()),
        "per_class_IoU": [float(v) for v in iou.cpu().tolist()],
        "pixel_accuracy": float(correct / max(1, total)),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    dataset = build_dataset(args.split, args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    amp_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"
    target_size = (args.resize_height, args.resize_width)
    confusion = torch.zeros(args.num_classes, args.num_classes, dtype=torch.int64)
    correct = 0
    total = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()

    progress = tqdm(loader, desc=f"eval {args.split} {args.model_type} tta")
    with torch.no_grad():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                logits = tta_forward(
                    model,
                    args.model_type,
                    images,
                    args.num_classes,
                    target_size,
                    args.scales,
                    args.hflip,
                )
            preds = logits.argmax(dim=1).cpu()
            batch_correct, batch_total = update_confusion(
                confusion,
                preds,
                labels,
                args.num_classes,
                args.ignore_index,
            )
            correct += batch_correct
            total += batch_total
            current = metrics_from_confusion(confusion, correct, total)
            progress.set_postfix(miou=f"{current['mIoU']:.4f}", acc=f"{current['pixel_accuracy']:.4f}")

    synchronize(device)
    elapsed = time.perf_counter() - start
    metrics = metrics_from_confusion(confusion, correct, total)
    num_images = len(dataset)
    tta_forwards = len(args.scales) * (2 if args.hflip else 1)
    class_names = class_names_for(args.num_classes)
    metrics.update(
        {
            "model_type": args.model_type,
            "checkpoint": args.checkpoint,
            "student_model": args.student_model if args.model_type == "student" else None,
            "split": args.split,
            "num_images": num_images,
            "num_classes": args.num_classes,
            "class_names": class_names,
            "per_class_IoU_named": [
                {
                    "class_id": class_id,
                    "class_name": class_names[class_id],
                    "iou": metrics["per_class_IoU"][class_id],
                }
                for class_id in range(args.num_classes)
            ],
            "input_size": [args.resize_height, args.resize_width],
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "scales": args.scales,
            "hflip": args.hflip,
            "tta_forwards_per_image": tta_forwards,
            "elapsed_sec": elapsed,
            "latency_ms_per_image_dataset": elapsed * 1000.0 / max(1, num_images),
            "throughput_images_per_sec_dataset": num_images / max(elapsed, 1e-12),
            "parameters": count_parameters(model),
            "model_size_mb": model_size_mb(model),
            "peak_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            ),
        }
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
