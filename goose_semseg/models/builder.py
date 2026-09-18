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


class SpectralInputAdapter(torch.nn.Module):
    """Learn a multispectral-to-RGB representation for an RGB-pretrained backbone."""

    def __init__(self, in_channels: int, hidden_channels: int = 16, add_indices: bool = False,
                 precomputed_indices: bool = False):
        super().__init__()
        if in_channels < 3:
            raise ValueError("SpectralInputAdapter requires at least three channels.")
        self.add_indices = add_indices
        self.precomputed_indices = precomputed_indices
        adapter_channels = (in_channels if add_indices else in_channels - 2) if precomputed_indices else in_channels + 2 * int(add_indices)
        self.rgb_projection = torch.nn.Conv2d(adapter_channels, 3, kernel_size=1, bias=False)
        self.spectral_residual = torch.nn.Sequential(
            torch.nn.Conv2d(adapter_channels, hidden_channels, kernel_size=1),
            torch.nn.GELU(),
            torch.nn.Conv2d(hidden_channels, 3, kernel_size=1),
        )
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )
        with torch.no_grad():
            self.rgb_projection.weight.zero_()
            # Input order is Blue, Green, Red, NIR, RedEdge, Thermal.
            self.rgb_projection.weight[0, 2, 0, 0] = 1.0
            self.rgb_projection.weight[1, 1, 0, 0] = 1.0
            self.rgb_projection.weight[2, 0, 0, 0] = 1.0
            self.spectral_residual[-1].weight.zero_()
            self.spectral_residual[-1].bias.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.precomputed_indices:
            if not self.add_indices:
                inputs = inputs[:, :-2]
        elif self.add_indices:
            red, nir, edge = inputs[:, 2:3], inputs[:, 3:4], inputs[:, 4:5]
            available = (inputs[:, 3:6].abs().sum((1,2,3),keepdim=True)>0).to(inputs.dtype)
            inputs = torch.cat((inputs, available*(nir-red)/(nir+red+1e-6),
                                available*(nir-edge)/(nir+edge+1e-6)),dim=1)
        rgb_like = self.rgb_projection(inputs) + self.spectral_residual(inputs)
        return (rgb_like - self.mean) / self.std


class SpectralFeatureFusion(torch.nn.Module):
    """Fuse extra spectral bands with DINOv3 features."""

    def __init__(self, backbone: torch.nn.Module, mode: str, embed_dim: int, spectral_channels: int = 3,
                 improved_fusion: bool = False):
        super().__init__()
        self.backbone = backbone
        self.mode = mode
        self.spectral_channels = spectral_channels
        self.improved_fusion = improved_fusion
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )
        self.spectral_stem = torch.nn.Sequential(
            torch.nn.Conv2d(spectral_channels, 64, kernel_size=3, stride=4 if improved_fusion else 1, padding=1, bias=False),
            torch.nn.GroupNorm(8, 64),
            torch.nn.GELU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(8, 64),
            torch.nn.GELU(),
        )
        if mode == "late":
            self.projections = torch.nn.ModuleDict({"4": torch.nn.Conv2d(64, embed_dim, 1, bias=False)})
        elif mode == "gated":
            self.projections = torch.nn.ModuleDict({
                level: torch.nn.Conv2d(64, embed_dim, 1, bias=False)
                for level in ("1", "2", "3", "4")
            })
            self.gates = torch.nn.ModuleDict({
                level: torch.nn.Conv2d(64 + (embed_dim if improved_fusion else 0), 1, 1)
                for level in ("1", "2", "3", "4")
            })
        elif mode == "cross_attention":
            self.spectral_projection = torch.nn.Conv2d(64, embed_dim, 1, bias=False)
            self.cross_attention = torch.nn.MultiheadAttention(
                embed_dim, num_heads=8, dropout=0.1, batch_first=True
            )
            self.cross_norm = torch.nn.LayerNorm(embed_dim)
            self.cross_scale = torch.nn.Parameter(torch.tensor(0.1))
        else:
            raise ValueError(f"Unsupported feature fusion mode: {mode}")
        if improved_fusion:
            # Begin at the RGB pretrained representation while fusion learns.
            with torch.no_grad():
                if mode in ('late', 'gated'):
                    for projection in self.projections.values():
                        projection.weight.zero_()
                else:
                    self.cross_scale.fill_(0.0)

    def _rgb_input(self, inputs: torch.Tensor) -> torch.Tensor:
        rgb = torch.stack((inputs[:, 2], inputs[:, 1], inputs[:, 0]), dim=1)
        return (rgb - self.mean) / self.std

    def forward(self, inputs: torch.Tensor):
        backbone_outputs = self.backbone(self._rgb_input(inputs))
        cls_feature = None
        if isinstance(backbone_outputs, tuple):
            features, cls_feature = backbone_outputs
        else:
            features = backbone_outputs

        extra = inputs[:, 3:3+self.spectral_channels]
        spectral = self.spectral_stem(extra)
        has_spectral = (extra.abs().sum(dim=(1, 2, 3), keepdim=True) > 0).to(extra.dtype)
        fused = dict(features)

        if self.mode == "late":
            level = "4"
            spec = torch.nn.functional.interpolate(
                spectral, size=features[level].shape[-2:], mode="bilinear", align_corners=False
            )
            fused[level] = features[level] + has_spectral * self.projections[level](spec)
        elif self.mode == "gated":
            for level in ("1", "2", "3", "4"):
                spec = torch.nn.functional.interpolate(
                    spectral, size=features[level].shape[-2:], mode="bilinear", align_corners=False
                )
                gate_input = torch.cat((features[level],spec),dim=1) if self.improved_fusion else spec
                gate = torch.sigmoid(self.gates[level](gate_input))
                fused[level] = features[level] + has_spectral * gate * self.projections[level](spec)
        else:
            level = "4"
            pooled = torch.nn.functional.adaptive_avg_pool2d(spectral, (8, 8))
            spectral_tokens = self.spectral_projection(pooled).flatten(2).transpose(1, 2)
            rgb_feature = features[level]
            rgb_tokens = rgb_feature.flatten(2).transpose(1, 2)
            attended, _ = self.cross_attention(rgb_tokens, spectral_tokens, spectral_tokens)
            attended = self.cross_norm(attended).transpose(1, 2).reshape_as(rgb_feature)
            fused[level] = rgb_feature + has_spectral * self.cross_scale * attended

        return (fused, cls_feature) if cls_feature is not None else fused

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
                if isinstance(inputs, tuple):
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
                    if isinstance(out, tuple):
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
    input_channels: int = 3,
    fusion_type: str = "input",
    precomputed_indices: bool = False,
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
            # Adding an auxiliary head must not change subsequent segmentation
            # parameter initialization for a same-seed on/off comparison.
            with torch.random.fork_rng(devices=[]):
                cls_head = CLSMultiLabelHead(in_dim=embed_dim, num_labels=cls_aux_num_classes)
        modules = []
        if fusion_type in ("input", "indices"):
            if input_channels != 3:
                modules.append(SpectralInputAdapter(
                    input_channels, add_indices=(fusion_type == "indices"),
                    precomputed_indices=precomputed_indices,
                ))
            modules.append(backbone_model)
        elif fusion_type in ("late", "gated", "cross_attention"):
            channels = input_channels - 3 - (2 if precomputed_indices else 0)
            modules.append(SpectralFeatureFusion(backbone_model, fusion_type, embed_dim, channels,
                                                improved_fusion=precomputed_indices))
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")
        modules.extend([feature_aligner, decoder])
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
