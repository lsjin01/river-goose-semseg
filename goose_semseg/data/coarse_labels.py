from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch


# River-pollution taxonomy.
#
#   Segmentation (fine)  = 중분류, ids 0..11  (0 = background) -> num_classes = 12
#   CLS auxiliary (coarse) = 대분류, 3 classes (NO Void / NO background slot)
#
# The CLS aux multi-hot target is derived from the fine mask via
# GOOSE_FINE_TO_COARSE below; background (fine id 0) is intentionally NOT in the
# coarse mapping, so it never lights up any of the 3 대분류 slots.
GOOSE_COARSE_CATEGORIES = (
    "농업계",            # 0
    "축산계",            # 1
    "하천수면 및 수변",   # 2
)
GOOSE_COARSE_NAME_TO_ID = {
    name: index for index, name in enumerate(GOOSE_COARSE_CATEGORIES)
}
GOOSE_NUM_COARSE_CATEGORIES = len(GOOSE_COARSE_CATEGORIES)  # 3

# 중분류 (fine) class names. id 0 is background and has no 대분류; ids 1..11 are
# the 11 중분류 confirmed by a full scan of all 96,340 label JSONs.
GOOSE_FINE_CLASS_NAMES = {
    0: "background",
    1: "밭_논",
    2: "잔재물",
    3: "배수로",
    4: "비닐하우스",
    5: "과수원",
    6: "축사",
    7: "야적퇴비_가축분뇨",
    8: "목장",
    9: "분뇨개별처리시설",
    10: "부유쓰레기",
    11: "연못",
}

# 중분류 -> 대분류 grouping for CLS auxiliary learning.
# background is deliberately omitted (no coarse target for it).
GOOSE_CLASSNAME_TO_COARSE = {
    "밭_논": "농업계",
    "잔재물": "농업계",
    "배수로": "농업계",
    "비닐하우스": "농업계",
    "과수원": "농업계",
    "축사": "축산계",
    "야적퇴비_가축분뇨": "축산계",
    "목장": "축산계",
    "분뇨개별처리시설": "축산계",
    "부유쓰레기": "하천수면 및 수변",
    "연못": "하천수면 및 수변",
}


def build_fine_to_coarse_mapping() -> Dict[int, int]:
    fine_to_coarse: Dict[int, int] = {}
    for fine_id, class_name in GOOSE_FINE_CLASS_NAMES.items():
        coarse_name = GOOSE_CLASSNAME_TO_COARSE.get(class_name)
        if coarse_name is None:
            continue
        fine_to_coarse[fine_id] = GOOSE_COARSE_NAME_TO_ID[coarse_name]
    return fine_to_coarse


GOOSE_FINE_TO_COARSE = build_fine_to_coarse_mapping()


def build_fine_multi_hot(
    mask: torch.Tensor,
    num_classes: int = len(GOOSE_FINE_CLASS_NAMES),
    ignore_index: int = 255,
) -> torch.Tensor:
    target = torch.zeros(num_classes, dtype=torch.float32, device=mask.device)
    unique_labels = torch.unique(mask.long())
    valid_labels = unique_labels[
        (unique_labels != ignore_index) & (unique_labels >= 0) & (unique_labels < num_classes)
    ]
    if valid_labels.numel() > 0:
        target[valid_labels] = 1.0
    return target


def build_coarse_multi_hot(
    mask: torch.Tensor,
    fine_to_coarse: Optional[Dict[int, int]] = None,
    num_coarse: int = GOOSE_NUM_COARSE_CATEGORIES,
    ignore_index: int = 255,
) -> torch.Tensor:
    if fine_to_coarse is None:
        fine_to_coarse = GOOSE_FINE_TO_COARSE

    target = torch.zeros(num_coarse, dtype=torch.float32, device=mask.device)
    unique_labels = torch.unique(mask.long())
    unique_labels = unique_labels[unique_labels != ignore_index]

    for fine_id in unique_labels.tolist():
        coarse_id = fine_to_coarse.get(int(fine_id))
        if coarse_id is not None and 0 <= coarse_id < num_coarse:
            target[coarse_id] = 1.0

    return target


def build_batch_coarse_targets(
    masks: torch.Tensor,
    fine_to_coarse: Optional[Dict[int, int]] = None,
    num_coarse: int = GOOSE_NUM_COARSE_CATEGORIES,
    ignore_index: int = 255,
) -> torch.Tensor:
    if fine_to_coarse is None:
        fine_to_coarse = GOOSE_FINE_TO_COARSE

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks.unsqueeze(0)

    targets = [
        build_coarse_multi_hot(
            mask,
            fine_to_coarse=fine_to_coarse,
            num_coarse=num_coarse,
            ignore_index=ignore_index,
        )
        for mask in masks
    ]
    return torch.stack(targets, dim=0)


def build_batch_fine_targets(
    masks: torch.Tensor,
    num_classes: int = len(GOOSE_FINE_CLASS_NAMES),
    ignore_index: int = 255,
) -> torch.Tensor:
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    elif masks.ndim == 2:
        masks = masks.unsqueeze(0)

    targets = [
        build_fine_multi_hot(
            mask,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        for mask in masks
    ]
    return torch.stack(targets, dim=0)


def build_batch_cls_aux_targets(
    masks: torch.Tensor,
    *,
    target_type: str = "coarse",
    num_classes: int = len(GOOSE_FINE_CLASS_NAMES),
    fine_to_coarse: Optional[Dict[int, int]] = None,
    num_coarse: int = GOOSE_NUM_COARSE_CATEGORIES,
    ignore_index: int = 255,
) -> torch.Tensor:
    target_type = str(target_type).lower()
    if target_type == "coarse":
        return build_batch_coarse_targets(
            masks,
            fine_to_coarse=fine_to_coarse,
            num_coarse=num_coarse,
            ignore_index=ignore_index,
        )
    if target_type == "fine":
        return build_batch_fine_targets(
            masks,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    raise ValueError(f"Unsupported cls_aux target_type {target_type!r}. Expected 'coarse' or 'fine'.")
