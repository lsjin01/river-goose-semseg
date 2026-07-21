from __future__ import annotations

import torch.nn as nn
from torch.optim import AdamW


def build_optimizer(
    model: nn.Module,
    lr: float,
    encoder_lr: float,
    weight_decay: float,
) -> AdamW:
    encoder_parameters = []
    other_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if ".backbone." in name or name.startswith("segmentation_model.0.backbone"):
            encoder_parameters.append(parameter)
        else:
            other_parameters.append(parameter)

    parameter_groups = []
    if other_parameters:
        parameter_groups.append(
            {"params": other_parameters, "lr": lr, "weight_decay": weight_decay}
        )
    if encoder_parameters:
        parameter_groups.append(
            {"params": encoder_parameters, "lr": encoder_lr, "weight_decay": weight_decay}
        )
    return AdamW(parameter_groups)
