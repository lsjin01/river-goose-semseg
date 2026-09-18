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
    # Select the actual pretrained ViT by object identity, independent of wrapper names.
    from goose_semseg.models.backbone.adapter import DINOv3_Adapter
    encoder_ids = {id(p) for module in model.modules() if isinstance(module, DINOv3_Adapter)
                   for p in module.backbone.parameters()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in encoder_ids:
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
