from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR


def load_model_state_allowing_token_specialization(
    model: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], int]:
    target_state_dict = model.state_dict()
    adapted_state_dict: Dict[str, torch.Tensor] = {}
    used_source_keys = set()

    for target_key, target_value in target_state_dict.items():
        source_value = source_state_dict.get(target_key)
        if source_value is None:
            continue
        if source_value.shape != target_value.shape:
            continue
        adapted_state_dict[target_key] = source_value
        used_source_keys.add(target_key)

    load_result = model.load_state_dict(adapted_state_dict, strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = sorted(set(source_state_dict) - used_source_keys)
    return missing_keys, unexpected_keys, 0


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    scheduler: Optional[MultiStepLR],
    epoch: int,
    best_val_miou: float,
    args: argparse.Namespace,
) -> None:
    checkpoint_model = (
        model.module
        if isinstance(model, (nn.DataParallel, DistributedDataParallel))
        else model
    )
    payload = {
        "epoch": epoch,
        "best_val_miou": best_val_miou,
        # Keep checkpoints portable between single-GPU and DataParallel runs.
        "model_state_dict": checkpoint_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    # A training process may be interrupted while a multi-gigabyte checkpoint
    # is being written.  Write beside the destination and publish atomically so
    # the previous valid checkpoint is never replaced by a partial file.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)
