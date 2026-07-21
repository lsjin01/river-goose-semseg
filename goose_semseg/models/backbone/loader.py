from __future__ import annotations

import argparse
from typing import Sequence, Union
from urllib.error import HTTPError, URLError

import torch
import torch.nn as nn

from dinov3.hub.backbones import dinov3_vitl16


class HFBackboneIntermediateLayersAdapter(nn.Module):
    def __init__(self, model: nn.Module, embed_dim: int, patch_size: int):
        super().__init__()
        self.model = model
        self.config = getattr(model, "config", None)
        self.embed_dim = int(embed_dim)
        self.patch_size = int(patch_size)
        self.num_hidden_layers = int(getattr(self.config, "num_hidden_layers", 0))
        self.n_blocks = self.num_hidden_layers
        self.num_register_tokens = int(
            getattr(
                self.config,
                "num_register_tokens",
                getattr(self.config, "n_storage_tokens", 0) or 0,
            )
            or 0
        )
        self.untie_cls_and_patch_norms = bool(
            getattr(model, "untie_cls_and_patch_norms", False)
        )
        self.norm = getattr(model, "norm", None)
        self.cls_norm = getattr(model, "cls_norm", None)

    def _normalize_hidden_state(self, hidden_state: torch.Tensor) -> torch.Tensor:
        if self.untie_cls_and_patch_norms and self.cls_norm is not None and self.norm is not None:
            cls_and_registers = self.cls_norm(hidden_state[:, : self.num_register_tokens + 1])
            patch_tokens = self.norm(hidden_state[:, self.num_register_tokens + 1 :])
            return torch.cat((cls_and_registers, patch_tokens), dim=1)
        if self.norm is not None:
            return self.norm(hidden_state)
        return hidden_state

    def _resolve_layer_indexes(self, n: Union[int, Sequence[int]]) -> list[int]:
        if isinstance(n, int):
            if n <= 0 or n > self.num_hidden_layers:
                raise ValueError(
                    f"Requested {n} layers, but the HF backbone has {self.num_hidden_layers} blocks."
                )
            return list(range(self.num_hidden_layers - n, self.num_hidden_layers))
        layer_indexes = [int(index) for index in n]
        if not layer_indexes:
            raise ValueError("At least one intermediate layer index must be provided.")
        for index in layer_indexes:
            if index < 0 or index >= self.num_hidden_layers:
                raise ValueError(
                    f"Intermediate layer index {index} is out of range for {self.num_hidden_layers} blocks."
                )
        return layer_indexes

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        del reshape
        layer_indexes = self._resolve_layer_indexes(n)
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("HF DINOv3 backbone did not return hidden states.")

        selected_hidden_states = [hidden_states[index + 1] for index in layer_indexes]
        if norm:
            selected_hidden_states = [
                self._normalize_hidden_state(hidden_state)
                for hidden_state in selected_hidden_states
            ]

        class_tokens = [hidden_state[:, 0] for hidden_state in selected_hidden_states]
        patch_tokens = [
            hidden_state[:, self.num_register_tokens + 1 :]
            for hidden_state in selected_hidden_states
        ]
        if return_class_token:
            return tuple(zip(patch_tokens, class_tokens))
        return tuple(patch_tokens)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def load_hf_dinov3_backbone(model_name_or_path: str, local_files_only: bool) -> nn.Module:
    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    embed_dim = int(getattr(config, "hidden_size", getattr(model, "embed_dim", 0)))
    patch_size = getattr(config, "patch_size", 16)
    if isinstance(patch_size, (tuple, list)):
        patch_size = patch_size[0]
    return HFBackboneIntermediateLayersAdapter(model, embed_dim=embed_dim, patch_size=int(patch_size))


def load_dinov3_backbone(args: argparse.Namespace) -> nn.Module:
    loader = str(args.backbone_loader).lower()
    if loader in {"auto", "native"}:
        try:
            if args.dinov3_weights:
                return dinov3_vitl16(pretrained=True, weights=args.dinov3_weights)
            return dinov3_vitl16(pretrained=True)
        except (HTTPError, URLError, RuntimeError, OSError) as exc:
            if loader == "native":
                raise
            print(
                "Native DINOv3 weight loading failed; falling back to Hugging Face "
                f"{args.hf_dinov3_model_name_or_path}. Original error: {exc}"
            )

    return load_hf_dinov3_backbone(
        args.hf_dinov3_model_name_or_path,
        local_files_only=args.hf_local_files_only,
    )
