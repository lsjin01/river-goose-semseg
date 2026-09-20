#!/usr/bin/env python3
"""Distill the 9-channel seven-class teacher into a SegFormer student."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party")]

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from goose_semseg.data.merged_algae import CLASS_NAMES
from goose_semseg.data.spectral_tiles import OverlapSpectralTileDataset, SpectralTileDataset
from goose_semseg.utils.metrics import compute_mean_iou, per_class_metric_rows, update_confusion_matrix
from goose_semseg.utils.seed import seed_everything
from train_student_kd import (
    build_student,
    forward_student,
    kd_loss,
    load_teacher,
    mask2former_outputs_to_semantic_probs,
)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def clean_metric(matrix: torch.Tensor) -> dict:
    rows = per_class_metric_rows(matrix.cpu(), 0)
    for row in rows:
        row.pop("epoch", None)
        row["class_name"] = CLASS_NAMES[row["class_id"]]
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
    return {
        "miou_gt_supported": compute_mean_iou(matrix.cpu(), gt_present_only=True),
        "miou_union_present": compute_mean_iou(matrix.cpu()),
        "valid_pixels": int(matrix.sum()),
        "per_class": rows,
        "confusion_matrix": matrix.cpu().tolist(),
    }


@torch.inference_mode()
def overlap_validate(model, data_path: str, tile_size: int, stride: int, batch_size: int,
                     workers: int, device: torch.device, description: str) -> dict:
    dataset = OverlapSpectralTileDataset(data_path, "val", tile_size, stride)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                        pin_memory=True, prefetch_factor=1)
    matrices = {
        name: torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64, device=device)
        for name in ("overall", "Altum", "P1")
    }
    axis = torch.arange(tile_size, device=device, dtype=torch.float32) + 0.5
    sine = torch.sin(torch.pi * axis / tile_size)
    blend = (0.1 + 0.9 * torch.outer(sine, sine)).unsqueeze(0)
    current_frame = None
    score_sum = weight_sum = None
    offset = completed = 0
    model.eval()

    def finalize(frame_index: int) -> None:
        nonlocal score_sum, weight_sum, completed
        record = dataset.records[frame_index]
        if torch.any(weight_sum <= 0):
            raise RuntimeError(f"Incomplete overlap coverage for val frame {frame_index}")
        prediction = (score_sum / weight_sum).argmax(0)
        target = torch.from_numpy(dataset._mask(frame_index)).to(device=device, dtype=torch.int64)
        update_confusion_matrix(matrices["overall"], prediction, target, len(CLASS_NAMES), 255)
        update_confusion_matrix(matrices[record["sensor"]], prediction, target, len(CLASS_NAMES), 255)
        score_sum = weight_sum = None
        completed += 1

    progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=False)
    for images, _ in progress:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_student(model, images, (tile_size, tile_size))
            scores = logits.softmax(1).float()
        for batch_index in range(len(images)):
            frame_index, left, top = dataset.tiles[offset + batch_index]
            if current_frame != frame_index:
                if current_frame is not None:
                    finalize(current_frame)
                current_frame = frame_index
                record = dataset.records[frame_index]
                score_sum = torch.zeros(
                    (len(CLASS_NAMES), record["height"], record["width"]),
                    dtype=torch.float32, device=device,
                )
                weight_sum = torch.zeros(
                    (1, record["height"], record["width"]),
                    dtype=torch.float32, device=device,
                )
            score_sum[:, top:top + tile_size, left:left + tile_size].add_(
                scores[batch_index] * blend
            )
            weight_sum[:, top:top + tile_size, left:left + tile_size].add_(blend)
        offset += len(images)
        if offset % 1000 < len(images):
            progress.set_postfix(frames=completed)
    if current_frame is not None:
        finalize(current_frame)
    if offset != len(dataset) or completed != len(dataset.records):
        raise RuntimeError("Incomplete overlap validation traversal")
    if not torch.equal(matrices["overall"], matrices["Altum"] + matrices["P1"]):
        raise RuntimeError("Sensor confusion matrices do not sum to overall")
    return {name: clean_metric(matrix) for name, matrix in matrices.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-model", choices=("segformer_b0", "segformer_b1"), required=True)
    parser.add_argument("--data-path", default="data/labeling_merged_algae_7class_v2_geo")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=768)
    parser.add_argument("--val-stride", type=int, default=512)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-updates", type=int, default=200)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--kd-temperature", type=float, default=4.0)
    parser.add_argument("--use-dice", action="store_true")
    parser.add_argument("--class-sampling-prob", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--student-local-files-only", action="store_true", default=True)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def dice_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels != 255
    safe = labels.masked_fill(~valid, 0)
    target = F.one_hot(safe, len(CLASS_NAMES)).permute(0, 3, 1, 2).float() * valid[:, None]
    probs = logits.softmax(1) * valid[:, None]
    intersection = (probs * target).sum((0, 2, 3))
    cardinality = probs.sum((0, 2, 3)) + target.sum((0, 2, 3))
    return 1.0 - ((2 * intersection + 1) / (cardinality + 1)).mean()


def main() -> None:
    args = parse_args()
    # Persist the fixed input contract so student evaluators can reconstruct the wrapper.
    args.input_channels = 9
    args.source_tiles = True
    args.precomputed_indices = True
    args.num_classes = len(CLASS_NAMES)
    args.ignore_index = 255
    seed_everything(args.seed)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "train_args.json", vars(args))
    device = torch.device("cuda")

    train_set = SpectralTileDataset(
        args.data_path, "train", args.tile_size, flip_prob=0.5,
        rare_prob=args.class_sampling_prob, class_sampling_mode="within_image",
    )
    if train_set.manifest.get("class_names") != list(CLASS_NAMES):
        raise ValueError("Dataset taxonomy/order does not match merged_algae_7class")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=1,
    )
    student = build_student(
        args.student_model, len(CLASS_NAMES), args.student_local_files_only,
        input_channels=9, precomputed_indices=True,
    ).to(device)

    teacher_args = argparse.Namespace(
        teacher_checkpoint=args.teacher_checkpoint,
        teacher_model_factory="goose",
        teacher_strict_load=True,
        backbone_loader="hf",
        hf_dinov3_model_name_or_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
        hf_local_files_only=True,
        dinov3_weights=None,
        mask2former_pretrained_model_name_or_path="facebook/mask2former-swin-large-ade-semantic",
        disable_mask2former_pretrained=False,
        hidden_dim=256,
        num_classes=len(CLASS_NAMES),
        enable_cls_aux=False,
        cls_aux_num_classes=0,
        input_channels=9,
        fusion_type="input",
        source_tiles=True,
    )
    teacher = load_teacher(teacher_args, device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs

    def lr_factor(update: int) -> float:
        if update < args.warmup_updates:
            return max(1e-3, (update + 1) / max(1, args.warmup_updates))
        progress = (update - args.warmup_updates) / max(1, total_updates - args.warmup_updates)
        return max(0.0, 1.0 - progress) ** 0.9

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best_miou, stale_checks, update = -1.0, 0, 0
    metrics_path = output / "epoch_metrics.csv"
    with metrics_path.open("w", newline="") as handle:
        csv.writer(handle).writerow(
            ["epoch", "train_loss", "train_gt_loss", "train_kd_loss", "train_miou",
             "val_miou", "best_val_miou", "lr"]
        )

    for epoch in range(1, args.epochs + 1):
        student.train()
        confusion = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.int64, device=device)
        totals = {"loss": 0.0, "gt": 0.0, "kd": 0.0}
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} train", dynamic_ncols=True)
        for step, (images, labels) in enumerate(progress, 1):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                student_logits = forward_student(student, images, labels.shape[-2:])
                gt = F.cross_entropy(student_logits, labels, ignore_index=255)
                if args.use_dice:
                    gt = gt + dice_loss(student_logits, labels)
                with torch.no_grad():
                    teacher_probs = mask2former_outputs_to_semantic_probs(
                        teacher(images), len(CLASS_NAMES), labels.shape[-2:], args.kd_temperature
                    )
                kd = kd_loss(
                    student_logits, teacher_probs, args.kd_temperature, False, 0.0,
                    valid_mask=(labels != 255),
                )
                loss = gt + args.lambda_kd * kd
            (loss / args.grad_accum_steps).backward()
            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
            totals["loss"] += float(loss.detach())
            totals["gt"] += float(gt.detach())
            totals["kd"] += float(kd.detach())
            update_confusion_matrix(confusion, student_logits.argmax(1), labels, len(CLASS_NAMES), 255)
            train_miou = compute_mean_iou(confusion.cpu(), gt_present_only=True)
            progress.set_postfix(loss=f"{totals['loss']/step:.3f}", miou=f"{train_miou:.3f}")
            if args.print_every and (step == 1 or step % args.print_every == 0):
                print(json.dumps({"event": "train_step", "model": args.student_model,
                                  "epoch": epoch, "step": step, "steps": len(train_loader),
                                  "loss": float(loss.detach()), "miou": train_miou}), flush=True)

        val_miou = None
        if epoch % args.val_interval == 0:
            metrics = overlap_validate(
                student, args.data_path, args.tile_size, args.val_stride,
                args.val_batch_size, args.num_workers, device,
                f"Epoch {epoch}/{args.epochs} overlap-val",
            )
            val_miou = metrics["overall"]["miou_gt_supported"]
            atomic_json(output / f"val_epoch_{epoch:03d}.json", metrics)
            if val_miou > best_miou + 1e-4:
                best_miou, stale_checks = val_miou, 0
                torch.save({"model": student.state_dict(), "args": vars(args),
                            "epoch": epoch, "val_metrics": metrics}, output / "best.pt")
            else:
                stale_checks += 1

        denom = max(1, len(train_loader))
        row = [epoch, totals["loss"]/denom, totals["gt"]/denom, totals["kd"]/denom,
               train_miou, val_miou, best_miou, optimizer.param_groups[0]["lr"]]
        with metrics_path.open("a", newline="") as handle:
            csv.writer(handle).writerow(row)
        torch.save({"model": student.state_dict(), "args": vars(args), "epoch": epoch},
                   output / "latest.pt")
        atomic_json(output / "progress.json", {
            "status": "running", "student_model": args.student_model, "epoch": epoch,
            "train_miou": train_miou, "val_miou": val_miou,
            "best_val_miou": best_miou, "stale_validation_checks": stale_checks,
        })
        print(json.dumps({"event": "epoch", "epoch": epoch, "train_miou": train_miou,
                          "val_miou": val_miou, "best_val_miou": best_miou}), flush=True)
        if val_miou is not None and stale_checks >= args.early_stopping_patience:
            print(f"Early stopping after {stale_checks} validation checks", flush=True)
            break

    atomic_json(output / "progress.json", {
        "status": "completed", "student_model": args.student_model,
        "epoch": epoch, "best_val_miou": best_miou,
    })


if __name__ == "__main__":
    main()
