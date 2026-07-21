from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
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
    payload = {
        "epoch": epoch,
        "best_val_miou": best_val_miou,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "args": vars(args),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)
