from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def mask2former_semantic_scores(
    outputs: Dict[str, torch.Tensor],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    class_probs = F.softmax(outputs["pred_logits"].float(), dim=-1)[..., :-1]
    mask_probs = torch.sigmoid(outputs["pred_masks"].float())
    semantic_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    if semantic_logits.shape[-2:] != target_size:
        semantic_logits = F.interpolate(
            semantic_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return semantic_logits


def mask2former_to_semantic(
    outputs: Dict[str, torch.Tensor],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    return mask2former_semantic_scores(outputs, target_size=target_size).argmax(dim=1)
