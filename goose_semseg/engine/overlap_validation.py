"""Full-frame overlap validation shared by DDP ranks."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from goose_semseg.losses.m2f_criterion import Mask2FormerSetCriterion
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.utils.logging import update_loss_breakdown, zero_breakdown, zero_cls_metrics
from goose_semseg.utils.metrics import compute_mean_iou, update_confusion_matrix


def _distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def evaluate_overlap(
    *,
    model: torch.nn.Module,
    dataset,
    criterion: Mask2FormerSetCriterion,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    num_classes: int,
    ignore_index: int,
    tile_size: int,
    epoch: int,
    epochs: int,
) -> Tuple[float, float, Dict[str, float], Dict[str, float], torch.Tensor, Dict[str, torch.Tensor]]:
    """Blend semantic scores per frame, then reduce metrics across DDP ranks."""
    rank = dist.get_rank() if _distributed() else 0
    inference_model = model.module if isinstance(model, DistributedDataParallel) else model
    inference_model.eval()
    criterion.eval()
    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    axis = torch.arange(tile_size, device=device, dtype=torch.float32) + 0.5
    sine = torch.sin(torch.pi * axis / tile_size)
    blend = (0.1 + 0.9 * torch.outer(sine, sine)).unsqueeze(0)
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    group_names = sorted(
        {f"sensor/{record['sensor']}" for record in dataset.all_records}
        | {f"task/{record['group']}" for record in dataset.all_records}
    )
    group_matrices = {
        name: torch.zeros_like(matrix) for name in group_names
    }
    total_loss = 0.0
    total_batches = 0
    total_breakdown = zero_breakdown()
    current_frame = None
    score_sum = weight_sum = None
    offset = 0

    def finalize(frame_index: int) -> None:
        nonlocal score_sum, weight_sum
        record = dataset.records[frame_index]
        if torch.any(weight_sum <= 0):
            raise RuntimeError(f'Incomplete overlap coverage for frame {frame_index}')
        prediction = (score_sum / weight_sum).argmax(0)
        target = torch.from_numpy(dataset._mask(frame_index)).to(device=device, dtype=torch.int64)
        update_confusion_matrix(matrix, prediction, target, num_classes, ignore_index)
        for group in (f"sensor/{record['sensor']}", f"task/{record['group']}"):
            update_confusion_matrix(
                group_matrices[group], prediction, target, num_classes, ignore_index
            )
        score_sum = weight_sum = None

    progress = tqdm(
        loader,
        desc=f"Epoch {epoch + 1}/{epochs} overlap-val rank{rank}",
        dynamic_ncols=True,
        leave=False,
        disable=rank != 0,
    )
    with torch.inference_mode():
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = inference_model(images)
                loss, loss_dict = criterion(outputs, labels)
                scores = mask2former_semantic_scores(
                    outputs, target_size=(tile_size, tile_size)
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite overlap validation loss at epoch {epoch + 1}')
            total_loss += float(loss.item())
            total_batches += 1
            update_loss_breakdown(total_breakdown, loss_dict, criterion)
            for batch_index in range(len(images)):
                frame_index, left, top = dataset.tiles[offset + batch_index]
                if current_frame != frame_index:
                    if current_frame is not None:
                        finalize(current_frame)
                    current_frame = frame_index
                    record = dataset.records[frame_index]
                    score_sum = torch.zeros(
                        (num_classes, record['height'], record['width']),
                        dtype=torch.float32,
                        device=device,
                    )
                    weight_sum = torch.zeros(
                        (1, record['height'], record['width']),
                        dtype=torch.float32,
                        device=device,
                    )
                score_sum[:, top:top + tile_size, left:left + tile_size].add_(
                    scores[batch_index] * blend
                )
                weight_sum[:, top:top + tile_size, left:left + tile_size].add_(blend)
            offset += len(images)
            progress.set_postfix(loss=f'{total_loss / max(total_batches, 1):.4f}')
    if current_frame is not None:
        finalize(current_frame)
    if offset != len(dataset):
        raise RuntimeError('Incomplete overlap validation traversal')

    scalar_names = list(total_breakdown)
    scalars = torch.tensor(
        [total_loss, total_batches] + [total_breakdown[name] for name in scalar_names],
        dtype=torch.float64,
        device=device,
    )
    if _distributed():
        dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
        dist.all_reduce(matrix, op=dist.ReduceOp.SUM)
        for group_matrix in group_matrices.values():
            dist.all_reduce(group_matrix, op=dist.ReduceOp.SUM)
    global_batches = max(float(scalars[1].item()), 1.0)
    mean_loss = float(scalars[0].item() / global_batches)
    mean_breakdown = {
        name: float(scalars[index + 2].item() / global_batches)
        for index, name in enumerate(scalar_names)
    }
    mean_iou = compute_mean_iou(matrix, gt_present_only=True)
    return (
        mean_loss,
        mean_iou,
        mean_breakdown,
        zero_cls_metrics(),
        matrix.cpu(),
        {name: group_matrix.cpu() for name, group_matrix in group_matrices.items()},
    )
