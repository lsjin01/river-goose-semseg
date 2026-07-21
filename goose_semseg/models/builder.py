# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from enum import Enum
from functools import partial
from typing import Dict, Optional

import torch

from goose_semseg.models.backbone.adapter import DINOv3_Adapter
from goose_semseg.models.decoder.mask2former_head import Mask2FormerHead


class BackboneLayersSet(Enum):
    """
    Set of intermediate layers to take from the backbone.
    """

    LAST = "LAST"  # extracting only the last layer
    FOUR_LAST = "FOUR_LAST"  # extracting the four last layers
    FOUR_EVEN_INTERVALS = "FOUR_EVEN_INTERVALS"  # extracting outputs every 1/4 of the total number of blocks


def _get_backbone_out_indices(
    model: torch.nn.Module,
    backbone_out_layers: BackboneLayersSet = BackboneLayersSet.FOUR_EVEN_INTERVALS,
):
    """
    Get indices for output layers of the ViT backbone. For now there are 3 options available:
    BackboneLayersSet.LAST : only extract the last layer, used in segmentation tasks with a bn head.
    BackboneLayersSet.FOUR_EVEN_INTERVALS : extract outputs every 1/4 of the total number of blocks
    Reference outputs in 'FOUR_EVEN_INTERVALS' mode :
    ViT/S (12 blocks): [2, 5, 8, 11]
    ViT/B (12 blocks): [2, 5, 8, 11]
    ViT/L (24 blocks): [5, 11, 17, 23] (classic), [4, 11, 17, 23] (used in the paper)
    ViT/g (40 blocks): [9, 19, 29, 39]
    """
    n_blocks = getattr(model, "n_blocks", 1)
    if backbone_out_layers == BackboneLayersSet.LAST:
        out_indices = [n_blocks - 1]
    elif backbone_out_layers == BackboneLayersSet.FOUR_LAST:
        out_indices = [i for i in range(n_blocks - 4, n_blocks)]
    elif backbone_out_layers == BackboneLayersSet.FOUR_EVEN_INTERVALS:
        # Take indices that were used in the paper (for ViT/L only)
        if n_blocks == 24:
            out_indices = [4, 11, 17, 23]
        else:
            out_indices = [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    assert all([out_index < n_blocks for out_index in out_indices])
    return out_indices


def build_feature_channel_aligner(in_channels: int, out_channels: int) -> torch.nn.Module:
    if in_channels == out_channels:
        return torch.nn.Identity()

    aligner = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
    with torch.no_grad():
        aligner.weight.zero_()
        aligner.bias.zero_()
        for out_index in range(out_channels):
            in_index = min((out_index * in_channels) // out_channels, in_channels - 1)
            aligner.weight[out_index, in_index, 0, 0] = 1.0
    return aligner


class FeatureAligner(torch.nn.Module):
    def __init__(self, aligners: Dict[str, torch.nn.Module]):
        super().__init__()
        self.aligners = torch.nn.ModuleDict(aligners)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {level: self.aligners[level](features[level]) for level in ("1", "2", "3", "4")}


class CLSMultiLabelHead(torch.nn.Module):
    def __init__(self, in_dim: int, num_labels: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, num_labels)

    def forward(self, cls_feat: torch.Tensor) -> torch.Tensor:
        return self.fc(cls_feat)


class FeatureDecoder(torch.nn.Module):
    def __init__(
        self,
        segmentation_model: torch.nn.ModuleList,
        autocast_ctx,
        cls_head: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.segmentation_model = segmentation_model
        self.autocast_ctx = autocast_ctx
        self.cls_head = cls_head

    def forward(self, inputs):
        with self.autocast_ctx():
            cls_feat = None
            for module_index, module in enumerate(self.segmentation_model):
                inputs = module.forward(inputs)
                if module_index == 0 and isinstance(inputs, tuple):
                    inputs, cls_feat = inputs
            if self.cls_head is None:
                return inputs
            if cls_feat is None:
                raise ValueError("CLS auxiliary head requires the backbone to return a CLS feature.")
            outputs = dict(inputs)
            outputs["cls_logits"] = self.cls_head(cls_feat)
            return outputs

    def predict(self, inputs, rescale_to=(512, 512)):
        with torch.inference_mode():
            with self.autocast_ctx():
                out = inputs
                for module_index, module in enumerate(self.segmentation_model[:-1]):
                    out = module(out)
                    if module_index == 0 and isinstance(out, tuple):
                        out, _ = out
                out = self.segmentation_model[-1].predict(out, rescale_to=rescale_to)
        return out


def build_segmentation_decoder(
    backbone_model,
    backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
    decoder_type="linear",
    hidden_dim=2048,
    num_classes=150,
    dropout=0.1,
    autocast_dtype=torch.float32,
    freeze_backbone=True,
    feature_channels: Optional[Dict[str, int]] = None,
    cls_aux_num_classes: int = 0,
):
    backbone_indices_to_use = _get_backbone_out_indices(backbone_model, backbone_out_layers)
    autocast_ctx = partial(torch.autocast, device_type="cuda", enabled=True, dtype=autocast_dtype)
    cls_head = None
    if decoder_type == "m2f":
        backbone_model = DINOv3_Adapter(
            backbone_model,
            interaction_indexes=backbone_indices_to_use,
            freeze_backbone=freeze_backbone,
            return_cls_token=cls_aux_num_classes > 0,
        )
        backbone_model.eval()
        embed_dim = backbone_model.backbone.embed_dim
        patch_size = backbone_model.patch_size
        head_input_channels = feature_channels or {
            "1": embed_dim,
            "2": embed_dim,
            "3": embed_dim,
            "4": embed_dim,
        }
        feature_aligner = FeatureAligner(
            {
                level: build_feature_channel_aligner(embed_dim, head_input_channels[level])
                for level in ("1", "2", "3", "4")
            }
        )
        decoder = Mask2FormerHead(
            input_shape={
                "1": [head_input_channels["1"], patch_size * 4, patch_size * 4, 4],
                "2": [head_input_channels["2"], patch_size * 2, patch_size * 2, 8],
                "3": [head_input_channels["3"], patch_size, patch_size, 16],
                "4": [head_input_channels["4"], int(patch_size / 2), int(patch_size / 2), 32],
            },
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            ignore_value=255,
        )
        if cls_aux_num_classes > 0:
            cls_head = CLSMultiLabelHead(in_dim=embed_dim, num_labels=cls_aux_num_classes)
        modules = [backbone_model, feature_aligner, decoder]
    else:
        raise ValueError(
            f'Unsupported decoder "{decoder_type}". This minimal repo only supports "m2f".'
        )

    segmentation_model = FeatureDecoder(
        torch.nn.ModuleList(modules),
        autocast_ctx=autocast_ctx,
        cls_head=cls_head,
    )
    return segmentation_model
