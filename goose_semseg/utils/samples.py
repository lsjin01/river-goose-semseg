from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from goose_semseg import CLS_AUX_COARSE_NUM_CLASSES
from goose_semseg.data.coarse_labels import build_batch_cls_aux_targets


def _load_yaml_config(path: str) -> Dict[str, object]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in config file {config_path}")
    return data


def _resolve_cls_aux_output_dim(
    *,
    target_type: str,
    num_classes: int,
    taxonomy: str = 'legacy',
) -> int:
    target_type = str(target_type).lower()
    if target_type == "coarse":
        if taxonomy == 'labeling_data':
            from goose_semseg.data.labeling_taxonomy import COARSE_NAMES, FINE_NAMES
            if num_classes != len(FINE_NAMES):
                raise ValueError('labeling_data coarse auxiliary requires 11 fine classes')
            return len(COARSE_NAMES)
        if taxonomy != 'legacy':
            raise ValueError(f'Unknown CLS taxonomy: {taxonomy}')
        return CLS_AUX_COARSE_NUM_CLASSES
    if target_type == "fine":
        return int(num_classes)
    raise ValueError(f"Unsupported cls_aux_target_type {target_type!r}. Expected 'coarse' or 'fine'.")


def _compute_cls_aux_pos_weight_from_samples(
    *,
    samples: Sequence[Tuple[Path, Path]],
    target_type: str,
    num_classes: int,
    cls_aux_num_classes: int,
    ignore_index: int,
    max_pos_weight: float,
    fine_to_coarse: Optional[Dict[int, int]] = None,
) -> torch.Tensor:
    positive_counts = torch.zeros(cls_aux_num_classes, dtype=torch.float64)
    total_samples = 0

    for _, label_path in samples:
        label = np.array(Image.open(label_path), dtype=np.int64, copy=False)
        label_tensor = torch.from_numpy(label).unsqueeze(0)
        target = build_batch_cls_aux_targets(
            label_tensor,
            target_type=target_type,
            num_classes=num_classes,
            num_coarse=cls_aux_num_classes,
            ignore_index=ignore_index,
            fine_to_coarse=fine_to_coarse,
        )[0].to(dtype=torch.float64)
        positive_counts += target
        total_samples += 1

    if total_samples <= 0:
        raise ValueError("Cannot compute cls_aux pos_weight without training samples.")

    negatives = float(total_samples) - positive_counts
    positive_counts = positive_counts.clamp(min=1.0)
    pos_weight = negatives / positive_counts
    pos_weight = pos_weight.clamp(min=1.0, max=float(max_pos_weight))
    return pos_weight.to(dtype=torch.float32)
