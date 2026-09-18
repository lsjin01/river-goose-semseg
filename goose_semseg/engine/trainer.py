from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from goose_semseg.data.coarse_labels import build_batch_cls_aux_targets
from goose_semseg.losses.m2f_criterion import Mask2FormerSetCriterion
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.utils.logging import (
    update_loss_breakdown,
    zero_breakdown,
    zero_cls_metrics,
)
from goose_semseg.utils.metrics import compute_mean_iou, update_confusion_matrix


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    criterion: Mask2FormerSetCriterion,
    optimizer: Optional[AdamW],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: bool,
    grad_clip_norm: float,
    grad_accum_steps: int,
    num_classes: int,
    ignore_index: int,
    epoch: int,
    epochs: int,
    enable_cls_aux: bool,
    cls_aux_target_type: str,
    cls_aux_loss_type: str,
    cls_aux_weight: float,
    cls_aux_num_classes: int,
    cls_aux_pos_weight: Optional[torch.Tensor],
    iter_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    gt_present_only: bool = False,
    group_confusion_matrices: Optional[Dict[str, torch.Tensor]] = None,
    cls_aux_fine_to_coarse: Optional[Dict[int, int]] = None,
) -> Tuple[float, float, Dict[str, float], Dict[str, float], torch.Tensor]:
    is_train = optimizer is not None
    distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    model.train(is_train)
    criterion.train(is_train)

    total_loss = 0.0
    total_batches = 0
    total_breakdown = zero_breakdown()
    cls_correct = 0.0
    cls_count = 0.0
    confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    sample_offset = 0
    if group_confusion_matrices is not None:
        from torch.utils.data import SequentialSampler
        if is_train or not isinstance(loader.sampler, SequentialSampler) or not hasattr(loader.dataset, 'tiles'):
            raise ValueError('Group diagnostics require sequential, unshuffled source-tile evaluation')
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} {'train' if is_train else 'val'}",
        dynamic_ncols=True,
        leave=False,
        disable=rank != 0,
    )

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    if enable_cls_aux and cls_aux_loss_type == "weighted_bce":
        if cls_aux_pos_weight is None:
            raise ValueError("weighted_bce for cls_aux requires cls_aux_pos_weight.")
        cls_aux_pos_weight = cls_aux_pos_weight.to(device=device, non_blocking=True)

    for step, (images, labels) in enumerate(progress, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=amp and device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, labels)
                if enable_cls_aux:
                    cls_logits = outputs.get("cls_logits")
                    if cls_logits is None:
                        raise ValueError("CLS auxiliary supervision is enabled, but the model did not return cls_logits.")
                    cls_targets = build_batch_cls_aux_targets(
                        labels,
                        target_type=cls_aux_target_type,
                        num_classes=num_classes,
                        num_coarse=cls_aux_num_classes,
                        ignore_index=ignore_index,
                        fine_to_coarse=cls_aux_fine_to_coarse,
                    )
                    if cls_logits.shape[-1] != cls_targets.shape[-1]:
                        raise ValueError(
                            "CLS auxiliary head output dim does not match target dim: "
                            f"logits={cls_logits.shape[-1]} targets={cls_targets.shape[-1]} "
                            f"(target_type={cls_aux_target_type!r})."
                        )
                    raw_cls_loss = F.binary_cross_entropy_with_logits(
                        cls_logits.float(),
                        cls_targets,
                        pos_weight=cls_aux_pos_weight if cls_aux_loss_type == "weighted_bce" else None,
                    )
                    weighted_cls_loss = raw_cls_loss * cls_aux_weight
                    loss = loss + weighted_cls_loss
                    loss_dict = dict(loss_dict)
                    loss_dict["loss_cls_aux"] = weighted_cls_loss.detach()

                    cls_predictions = torch.sigmoid(cls_logits.float()) >= 0.5
                    cls_correct += float((cls_predictions == cls_targets.bool()).sum().item())
                    cls_count += float(cls_targets.numel())
                semantic_scores = mask2former_semantic_scores(
                    outputs,
                    target_size=labels.shape[-2:],
                )

        if not torch.isfinite(loss.detach()):
            raise FloatingPointError(f'Non-finite loss at epoch {epoch+1}, step {step}')

        if is_train:
            accumulation_start = ((step - 1) // grad_accum_steps) * grad_accum_steps
            accumulation_count = min(grad_accum_steps, len(loader) - accumulation_start)
            scaler.scale(loss / accumulation_count).backward()
            should_step = (step % grad_accum_steps == 0) or (step == len(loader))
            if should_step:
                scaler.unscale_(optimizer)
                params_to_clip = [
                    param
                    for group in optimizer.param_groups
                    for param in group["params"]
                    if param.grad is not None
                ]
                if params_to_clip:
                    torch.nn.utils.clip_grad_norm_(params_to_clip, grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if iter_scheduler is not None:
                    iter_scheduler.step()

        total_loss += float(loss.detach().item())
        total_batches += 1
        update_loss_breakdown(total_breakdown, loss_dict, criterion)

        predictions = semantic_scores.argmax(dim=1)
        update_confusion_matrix(confusion_matrix, predictions, labels, num_classes, ignore_index)
        if group_confusion_matrices is not None:
            for j in range(len(labels)):
                record = loader.dataset.records[loader.dataset.tiles[sample_offset+j][0]]
                for group in (f"sensor/{record['sensor']}", f"task/{record['group']}"):
                    if group not in group_confusion_matrices:
                        group_confusion_matrices[group] = torch.zeros_like(confusion_matrix)
                    update_confusion_matrix(group_confusion_matrices[group], predictions[j], labels[j],
                                            num_classes, ignore_index)
            sample_offset += len(labels)
        progress.set_postfix(
            loss=f"{total_loss / max(total_batches, 1):.4f}",
            miou=f"{compute_mean_iou(confusion_matrix, gt_present_only):.4f}",
        )

    scalar_names = list(total_breakdown)
    scalars = torch.tensor(
        [total_loss, total_batches, cls_correct, cls_count]
        + [total_breakdown[name] for name in scalar_names],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
    global_batches = max(float(scalars[1].item()), 1.0)
    mean_loss = float(scalars[0].item() / global_batches)
    mean_breakdown = {
        name: float(scalars[index + 4].item() / global_batches)
        for index, name in enumerate(scalar_names)
    }
    cls_metrics = zero_cls_metrics()
    global_cls_count = float(scalars[3].item())
    if enable_cls_aux and global_cls_count > 0.0:
        cls_metrics["cls_aux_accuracy"] = float(scalars[2].item()) / global_cls_count
    if group_confusion_matrices is not None:
        for group in group_confusion_matrices:
            group_confusion_matrices[group] = group_confusion_matrices[group].detach().cpu()
    return (
        mean_loss,
        compute_mean_iou(confusion_matrix, gt_present_only),
        mean_breakdown,
        cls_metrics,
        confusion_matrix.detach().cpu(),
    )
