#!/usr/bin/env python3
"""Train a lightweight SegFormer student with semantic-map distillation.

The teacher is expected to be an already trained Mask2Former-style model that
returns pred_logits and pred_masks. Query tensors are converted to dense
semantic probabilities before KD; query-to-query matching is intentionally not
used.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT, _ROOT / "third_party"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


STUDENT_CHECKPOINTS = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b1": "nvidia/segformer-b1-finetuned-ade-512-512",
}


class SegmentationFolderDataset(Dataset):
    """Simple image/mask folder dataset for KD experiments.

    This is intentionally small and dependency-light. If the project already
    has a dataset builder, pass it through --dataset_factory instead.
    """

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        image_size: int = 512,
        image_suffixes: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = image_size
        self.images = sorted(
            p for p in self.image_dir.iterdir() if p.suffix.lower() in image_suffixes
        )
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.images[idx]
        mask_path = self._find_mask(image_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        image_arr = np.asarray(image).astype(np.float32) / 255.0
        image_arr = (image_arr - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        mask_arr = np.asarray(mask).astype(np.int64)

        return {
            "pixel_values": torch.from_numpy(image_arr).permute(2, 0, 1),
            "labels": torch.from_numpy(mask_arr),
        }

    def _find_mask(self, image_path: Path) -> Path:
        candidates = [
            self.mask_dir / f"{image_path.stem}{suffix}"
            for suffix in (".png", ".tif", ".tiff", ".jpg", ".jpeg")
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No mask found for {image_path.name} in {self.mask_dir}")


def load_factory(factory: str):
    module_name, fn_name = factory.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def build_dataset(split: str, args: argparse.Namespace) -> Dataset:
    if args.dataset_factory:
        factory = load_factory(args.dataset_factory)
        return factory(split=split, args=args)

    if args.data_path:
        from goose_semseg.data.dataset import GooseSegmentationDataset

        resize_size = (args.resize_width, args.resize_height)
        if split == "train":
            return GooseSegmentationDataset(
                args.data_path,
                split,
                resize_size=resize_size,
                flip_prob=args.flip_prob,
                enable_random_crop=args.enable_random_crop,
                crop_size=(args.crop_width, args.crop_height),
                enable_rare_class_crop=args.enable_rare_class_crop,
                rare_class_crop_prob=args.rare_class_crop_prob,
                rare_class_ids=args.rare_class_ids,
                rare_class_min_pixels=args.rare_class_min_pixels,
                rare_class_min_ratio=args.rare_class_min_ratio,
                rare_class_crop_attempts=args.rare_class_crop_attempts,
            )
        return GooseSegmentationDataset(
            args.data_path,
            split,
            resize_size=resize_size,
            flip_prob=0.0,
        )

    if split == "train":
        image_dir, mask_dir = args.train_image_dir, args.train_mask_dir
    else:
        image_dir, mask_dir = args.val_image_dir, args.val_mask_dir
    if not image_dir or not mask_dir:
        raise ValueError(
            "Provide --dataset_factory or folder arguments: "
            "--train_image_dir/--train_mask_dir/--val_image_dir/--val_mask_dir"
        )
    return SegmentationFolderDataset(image_dir, mask_dir, image_size=args.image_size)


def build_student(student_model: str, num_classes: int, local_files_only: bool) -> nn.Module:
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise ImportError(
            "SegFormer student requires transformers. Install it or adapt build_student()."
        ) from exc

    checkpoint = STUDENT_CHECKPOINTS[student_model]
    return SegformerForSemanticSegmentation.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
        local_files_only=local_files_only,
    )


def _namespace_with_fallback(primary: Dict[str, Any], fallback: argparse.Namespace) -> argparse.Namespace:
    values = vars(fallback).copy()
    values.update(primary)
    return argparse.Namespace(**values)


def load_goose_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    from goose_semseg.models.backbone.loader import load_dinov3_backbone
    from goose_semseg.models.builder import BackboneLayersSet, build_segmentation_decoder
    from goose_semseg.pretrained.mask2former import (
        infer_pretrained_feature_channels,
        load_pretrained_mask2former,
    )
    from goose_semseg.utils.checkpoint import load_model_state_allowing_token_specialization

    checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
    teacher_args = _namespace_with_fallback(checkpoint.get("args", {}), args)
    teacher_args.device = str(device)

    backbone = load_dinov3_backbone(teacher_args)
    pretrained_mask2former = None
    feature_channels = None
    if not getattr(teacher_args, "disable_mask2former_pretrained", False):
        pretrained_mask2former = load_pretrained_mask2former(
            teacher_args.mask2former_pretrained_model_name_or_path
        )
        feature_channels = infer_pretrained_feature_channels(pretrained_mask2former)
        teacher_args.hidden_dim = int(pretrained_mask2former.config.hidden_dim)

    teacher = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type="m2f",
        hidden_dim=teacher_args.hidden_dim,
        num_classes=teacher_args.num_classes,
        autocast_dtype=torch.bfloat16,
        freeze_backbone=False,
        feature_channels=feature_channels,
        cls_aux_num_classes=(
            teacher_args.cls_aux_num_classes if getattr(teacher_args, "enable_cls_aux", False) else 0
        ),
    )
    missing_keys, unexpected_keys, _ = load_model_state_allowing_token_specialization(
        teacher,
        checkpoint["model_state_dict"],
    )
    if missing_keys:
        print(f"Warning: teacher load missing {len(missing_keys)} keys. First: {missing_keys[:5]}")
    if unexpected_keys:
        print(f"Warning: teacher load ignored {len(unexpected_keys)} keys. First: {unexpected_keys[:5]}")
    return teacher


def load_teacher(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.teacher_model_factory == "goose" or args.teacher_model_factory is None:
        teacher = load_goose_teacher(args, device)
    elif args.teacher_model_factory:
        teacher = load_factory(args.teacher_model_factory)(args=args)
        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
        state_dict = checkpoint.get(
            "model_state_dict",
            checkpoint.get("model", checkpoint.get("state_dict", checkpoint)),
        )
        teacher.load_state_dict(state_dict, strict=args.teacher_strict_load)
    else:
        checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu")
        if isinstance(checkpoint, nn.Module):
            teacher = checkpoint
        elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), nn.Module):
            teacher = checkpoint["model"]
        else:
            raise ValueError(
                "Could not reconstruct teacher from checkpoint alone. "
                "Pass --teacher_model_factory module:function."
            )

    teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def unpack_outputs(outputs: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs["pred_logits"], outputs["pred_masks"]
    return outputs.pred_logits, outputs.pred_masks


def mask2former_outputs_to_semantic_probs(
    outputs: Any,
    num_classes: int,
    target_size: Tuple[int, int],
    temperature: float,
) -> torch.Tensor:
    pred_logits, pred_masks = unpack_outputs(outputs)

    if pred_logits.shape[-1] > num_classes:
        class_probs = F.softmax(pred_logits / temperature, dim=-1)[..., :num_classes]
    else:
        class_probs = F.softmax(pred_logits[..., :num_classes] / temperature, dim=-1)
    mask_probs = pred_masks.sigmoid()

    semantic_probs = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    semantic_probs = F.interpolate(
        semantic_probs,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )
    semantic_probs = semantic_probs.clamp_min(1e-8)
    semantic_probs = semantic_probs / semantic_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return semantic_probs


def forward_student(student: nn.Module, pixel_values: torch.Tensor, target_size: Tuple[int, int]):
    outputs = student(pixel_values=pixel_values)
    logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
    if logits.shape[-2:] != target_size:
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
    return logits


def unpack_batch(batch: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        return batch["pixel_values"], batch["labels"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def dice_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    valid = labels != ignore_index
    labels_safe = labels.masked_fill(~valid, 0)
    one_hot = F.one_hot(labels_safe, num_classes=num_classes).permute(0, 3, 1, 2).float()
    probs = F.softmax(logits, dim=1)
    valid = valid.unsqueeze(1)
    probs = probs * valid
    one_hot = one_hot * valid
    dims = (0, 2, 3)
    intersection = (probs * one_hot).sum(dims)
    cardinality = probs.sum(dims) + one_hot.sum(dims)
    dice = (2.0 * intersection + 1.0) / (cardinality + 1.0)
    return 1.0 - dice.mean()


def kd_loss(
    student_logits: torch.Tensor,
    teacher_probs: torch.Tensor,
    temperature: float,
    enable_confidence_kd: bool,
    teacher_conf_threshold: float,
) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    per_class_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none")
    per_pixel_kl = per_class_kl.sum(dim=1)

    if not enable_confidence_kd:
        return per_pixel_kl.mean() * (temperature**2)

    teacher_conf = teacher_probs.max(dim=1).values
    weights = teacher_conf
    if teacher_conf_threshold > 0:
        weights = weights * (teacher_conf >= teacher_conf_threshold).float()
    denom = weights.sum().clamp_min(1.0)
    return ((per_pixel_kl * weights).sum() / denom) * (temperature**2)


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> None:
    valid = labels != ignore_index
    if not bool(valid.any()):
        return
    inds = num_classes * labels[valid].reshape(-1) + preds[valid].reshape(-1)
    confusion += torch.bincount(inds, minlength=num_classes**2).reshape(num_classes, num_classes)


def mean_iou_from_confusion(confusion: torch.Tensor) -> torch.Tensor:
    tp = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - tp
    return (tp / union.clamp_min(1.0)).mean()


@torch.no_grad()
def evaluate(
    student: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    epoch: int,
    epochs: int,
) -> Dict[str, Any]:
    student.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    correct = torch.tensor(0, dtype=torch.int64, device=device)
    total = torch.tensor(0, dtype=torch.int64, device=device)
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{epochs} val",
        dynamic_ncols=True,
        leave=False,
    )

    for batch in progress:
        pixel_values, labels = unpack_batch(batch)
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)
        logits = forward_student(student, pixel_values, labels.shape[-2:])
        preds = logits.argmax(dim=1)
        valid = labels != ignore_index
        correct += (preds[valid] == labels[valid]).sum()
        total += valid.sum()
        update_confusion_matrix(confusion, preds, labels, num_classes, ignore_index)
        progress.set_postfix(
            miou=f"{mean_iou_from_confusion(confusion).item():.4f}",
            acc=f"{(correct.float() / total.clamp_min(1).float()).item():.4f}",
        )

    tp = confusion.diag().float()
    union = confusion.sum(0).float() + confusion.sum(1).float() - tp
    iou = tp / union.clamp_min(1.0)
    return {
        "mIoU": iou.mean().item(),
        "per_class_IoU": iou.cpu().tolist(),
        "pixel_accuracy": (correct.float() / total.clamp_min(1).float()).item(),
    }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_checkpoint", required=True)
    parser.add_argument(
        "--teacher_model_factory",
        default="goose",
        help="Use 'goose' for this repo, or pass module:function for a custom teacher builder.",
    )
    parser.add_argument("--teacher_strict_load", action="store_true")
    parser.add_argument("--student_model", choices=sorted(STUDENT_CHECKPOINTS), default="segformer_b1")
    parser.add_argument("--student_local_files_only", action="store_true")
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--kd_temperature", type=float, default=4.0)
    parser.add_argument("--enable_confidence_kd", action="store_true")
    parser.add_argument("--teacher_conf_threshold", type=float, default=0.0)
    parser.add_argument("--use_dice", action="store_true")
    parser.add_argument("--dataset_factory", default=None)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--train_image_dir", default=None)
    parser.add_argument("--train_mask_dir", default=None)
    parser.add_argument("--val_image_dir", default=None)
    parser.add_argument("--val_mask_dir", default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--resize_width", type=int, default=512)
    parser.add_argument("--resize_height", type=int, default=512)
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
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument(
        "--print_every",
        type=int,
        default=0,
        help="Optional JSON train-step logs every N steps. Default 0 keeps tqdm-style output only.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--output_dir", default="outputs/student_kd_segformer_b1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
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


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "train_args.json", vars(args))

    device = torch.device(args.device)
    train_set = build_dataset("train", args)
    val_set = build_dataset("val", args)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    student = build_student(
        args.student_model,
        args.num_classes,
        local_files_only=args.student_local_files_only,
    ).to(device)
    teacher = None
    if args.lambda_kd > 0:
        teacher = load_teacher(args, device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_miou = -1.0

    print(f"Student parameters: {count_parameters(student):,}")
    print(f"Student model size MB: {model_size_mb(student):.2f}")

    for epoch in range(1, args.epochs + 1):
        student.train()
        running = {"total": 0.0, "gt": 0.0, "kd": 0.0}
        train_confusion = torch.zeros(
            (args.num_classes, args.num_classes),
            dtype=torch.int64,
            device=device,
        )
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{args.epochs} train",
            dynamic_ncols=True,
            leave=False,
        )

        for step, batch in enumerate(progress, start=1):
            pixel_values, labels = unpack_batch(batch)
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)
            target_size = labels.shape[-2:]

            student_logits = forward_student(student, pixel_values, target_size)
            gt_loss = F.cross_entropy(student_logits, labels, ignore_index=args.ignore_index)
            if args.use_dice:
                gt_loss = gt_loss + dice_loss(
                    student_logits, labels, args.num_classes, args.ignore_index
                )

            distill_loss = student_logits.new_tensor(0.0)
            if teacher is not None:
                with torch.no_grad():
                    teacher_outputs = teacher(pixel_values)
                    teacher_probs = mask2former_outputs_to_semantic_probs(
                        teacher_outputs,
                        num_classes=args.num_classes,
                        target_size=target_size,
                        temperature=args.kd_temperature,
                    )
                distill_loss = kd_loss(
                    student_logits,
                    teacher_probs,
                    temperature=args.kd_temperature,
                    enable_confidence_kd=args.enable_confidence_kd,
                    teacher_conf_threshold=args.teacher_conf_threshold,
                )

            loss = gt_loss + args.lambda_kd * distill_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running["total"] += loss.item()
            running["gt"] += gt_loss.item()
            running["kd"] += distill_loss.item()
            with torch.no_grad():
                preds = student_logits.argmax(dim=1)
                update_confusion_matrix(
                    train_confusion,
                    preds,
                    labels,
                    args.num_classes,
                    args.ignore_index,
                )
            seen = max(1, step)
            progress.set_postfix(
                loss=f"{running['total'] / seen:.4f}",
                gt=f"{running['gt'] / seen:.4f}",
                kd=f"{running['kd'] / seen:.4f}",
                miou=f"{mean_iou_from_confusion(train_confusion).item():.4f}",
            )
            if args.print_every > 0 and (step == 1 or step % args.print_every == 0):
                print(
                    json.dumps(
                        {
                            "event": "train_step",
                            "epoch": epoch,
                            "step": step,
                            "steps_per_epoch": len(train_loader),
                            "loss": float(loss.detach().item()),
                            "gt_loss": float(gt_loss.detach().item()),
                            "kd_loss": float(distill_loss.detach().item()),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        denom = max(1, len(train_loader))
        metrics = evaluate(
            student,
            val_loader,
            device,
            args.num_classes,
            args.ignore_index,
            epoch=epoch,
            epochs=args.epochs,
        )
        metrics.update(
            {
                "epoch": epoch,
                "train_total_loss": running["total"] / denom,
                "train_gt_loss": running["gt"] / denom,
                "train_kd_loss": running["kd"] / denom,
                "student_parameters": count_parameters(student),
                "student_model_size_mb": model_size_mb(student),
            }
        )
        print(json.dumps(metrics, ensure_ascii=False))
        save_json(output_dir / "latest_metrics.json", metrics)

        torch.save({"model": student.state_dict(), "args": vars(args), "metrics": metrics}, output_dir / "last.pt")
        if metrics["mIoU"] > best_miou:
            best_miou = metrics["mIoU"]
            torch.save(
                {"model": student.state_dict(), "args": vars(args), "metrics": metrics},
                output_dir / "best.pt",
            )


if __name__ == "__main__":
    main()
