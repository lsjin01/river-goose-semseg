from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn




def load_matching_state_dict(
    module: nn.Module,
    source_state_dict: Dict[str, torch.Tensor],
    module_name: str,
) -> None:
    target_state_dict = module.state_dict()
    matched_state_dict: Dict[str, torch.Tensor] = {}
    skipped_keys: list[str] = []

    for name, value in source_state_dict.items():
        target_value = target_state_dict.get(name)
        if target_value is None:
            skipped_keys.append(name)
            continue
        if target_value.shape != value.shape:
            skipped_keys.append(
                f"{name} (source={tuple(value.shape)}, target={tuple(target_value.shape)})"
            )
            continue
        matched_state_dict[name] = value

    missing_keys = sorted(set(target_state_dict) - set(matched_state_dict))
    module.load_state_dict(matched_state_dict, strict=False)

    print(
        f"Loaded {len(matched_state_dict)}/{len(target_state_dict)} "
        f"{module_name} tensors from the pretrained Mask2Former checkpoint."
    )
    if skipped_keys:
        preview = ", ".join(skipped_keys[:5])
        suffix = " ..." if len(skipped_keys) > 5 else ""
        print(
            f"Skipped {module_name} tensors due to shape/key mismatch: "
            f"{preview}{suffix}"
        )
    if missing_keys:
        preview = ", ".join(missing_keys[:5])
        suffix = " ..." if len(missing_keys) > 5 else ""
        print(f"Newly initialized {module_name} tensors: {preview}{suffix}")


def _remap_conv_norm_sequence_key(
    key: str,
    *,
    source_prefix: str,
    target_prefix: str,
) -> Optional[str]:
    if not key.startswith(source_prefix):
        return None

    remainder = key[len(source_prefix) :]
    parts = remainder.split(".")
    if len(parts) < 3:
        return None

    index, submodule = parts[0], parts[1]
    field = ".".join(parts[2:])
    if submodule == "0":
        return f"{target_prefix}{index}.{field}"
    if submodule == "1":
        return f"{target_prefix}{index}.norm.{field}"
    return None


def remap_local_pixel_decoder_state_dict(
    source_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}

    for name, value in source_state_dict.items():
        target_name: Optional[str] = None

        if name.startswith("input_projections."):
            target_name = name.replace("input_projections.", "input_convs.", 1)
        elif name.startswith("encoder.layers."):
            target_name = name.replace("encoder.layers.", "encoder.encoder.layers.", 1)
            target_name = target_name.replace(".self_attn_layer_norm.", ".norm1.")
            target_name = target_name.replace(".fc1.", ".linear1.")
            target_name = target_name.replace(".fc2.", ".linear2.")
            target_name = target_name.replace(".final_layer_norm.", ".norm2.")
        elif name == "level_embed":
            target_name = "encoder.level_encoding"
        elif name.startswith("mask_projection."):
            target_name = name.replace("mask_projection.", "mask_feature.", 1)
        elif name.startswith("lateral_convolutions."):
            target_name = _remap_conv_norm_sequence_key(
                name,
                source_prefix="lateral_convolutions.",
                target_prefix="lateral_convs.",
            )
        elif name.startswith("output_convolutions."):
            target_name = _remap_conv_norm_sequence_key(
                name,
                source_prefix="output_convolutions.",
                target_prefix="output_convs.",
            )

        if target_name is not None:
            remapped[target_name] = value

    return remapped


def remap_local_transformer_state_dict(
    transformer_state_dict: Dict[str, torch.Tensor],
    class_predictor_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    remapped: Dict[str, torch.Tensor] = {}

    direct_mappings = {
        "queries_features.weight": "query_feat.weight",
        "queries_embedder.weight": "query_embed.weight",
        "level_embed.weight": "level_embed.weight",
        "decoder.layernorm.weight": "post_norm.weight",
        "decoder.layernorm.bias": "post_norm.bias",
    }
    for source_name, target_name in direct_mappings.items():
        value = transformer_state_dict.get(source_name)
        if value is not None:
            remapped[target_name] = value

    for source_name, target_name in {
        "weight": "class_embed.weight",
        "bias": "class_embed.bias",
    }.items():
        value = class_predictor_state_dict.get(source_name)
        if value is not None:
            remapped[target_name] = value

    for index in range(3):
        for field in ("weight", "bias"):
            source_name = f"decoder.mask_predictor.mask_embedder.{index}.0.{field}"
            target_name = f"mask_embed.layers.{index}.{field}"
            value = transformer_state_dict.get(source_name)
            if value is not None:
                remapped[target_name] = value

    for index in range(3):
        for field in ("weight", "bias"):
            source_name = f"input_projections.{index}.{field}"
            target_name = f"input_proj.{index}.{field}"
            value = transformer_state_dict.get(source_name)
            if value is not None:
                remapped[target_name] = value

    decoder_layer_prefix = "decoder.layers."
    layer_indices = sorted(
        {
            int(key[len(decoder_layer_prefix) :].split(".", 1)[0])
            for key in transformer_state_dict
            if key.startswith(decoder_layer_prefix)
        }
    )

    for layer_index in layer_indices:
        source_prefix = f"decoder.layers.{layer_index}"

        q_proj_weight = transformer_state_dict.get(
            f"{source_prefix}.self_attn.q_proj.weight"
        )
        k_proj_weight = transformer_state_dict.get(
            f"{source_prefix}.self_attn.k_proj.weight"
        )
        v_proj_weight = transformer_state_dict.get(
            f"{source_prefix}.self_attn.v_proj.weight"
        )
        if (
            q_proj_weight is not None
            and k_proj_weight is not None
            and v_proj_weight is not None
        ):
            remapped[
                f"transformer_self_attention_layers.{layer_index}.self_attn.in_proj_weight"
            ] = torch.cat((q_proj_weight, k_proj_weight, v_proj_weight), dim=0)

        q_proj_bias = transformer_state_dict.get(
            f"{source_prefix}.self_attn.q_proj.bias"
        )
        k_proj_bias = transformer_state_dict.get(
            f"{source_prefix}.self_attn.k_proj.bias"
        )
        v_proj_bias = transformer_state_dict.get(
            f"{source_prefix}.self_attn.v_proj.bias"
        )
        if (
            q_proj_bias is not None
            and k_proj_bias is not None
            and v_proj_bias is not None
        ):
            remapped[
                f"transformer_self_attention_layers.{layer_index}.self_attn.in_proj_bias"
            ] = torch.cat((q_proj_bias, k_proj_bias, v_proj_bias), dim=0)

        for source_name, target_name in {
            "self_attn.out_proj.weight": "transformer_self_attention_layers.{i}.self_attn.out_proj.weight",
            "self_attn.out_proj.bias": "transformer_self_attention_layers.{i}.self_attn.out_proj.bias",
            "self_attn_layer_norm.weight": "transformer_self_attention_layers.{i}.norm.weight",
            "self_attn_layer_norm.bias": "transformer_self_attention_layers.{i}.norm.bias",
            "cross_attn.in_proj_weight": "transformer_cross_attention_layers.{i}.multihead_attn.in_proj_weight",
            "cross_attn.in_proj_bias": "transformer_cross_attention_layers.{i}.multihead_attn.in_proj_bias",
            "cross_attn.out_proj.weight": "transformer_cross_attention_layers.{i}.multihead_attn.out_proj.weight",
            "cross_attn.out_proj.bias": "transformer_cross_attention_layers.{i}.multihead_attn.out_proj.bias",
            "cross_attn_layer_norm.weight": "transformer_cross_attention_layers.{i}.norm.weight",
            "cross_attn_layer_norm.bias": "transformer_cross_attention_layers.{i}.norm.bias",
            "fc1.weight": "transformer_ffn_layers.{i}.linear1.weight",
            "fc1.bias": "transformer_ffn_layers.{i}.linear1.bias",
            "fc2.weight": "transformer_ffn_layers.{i}.linear2.weight",
            "fc2.bias": "transformer_ffn_layers.{i}.linear2.bias",
            "final_layer_norm.weight": "transformer_ffn_layers.{i}.norm.weight",
            "final_layer_norm.bias": "transformer_ffn_layers.{i}.norm.bias",
        }.items():
            value = transformer_state_dict.get(f"{source_prefix}.{source_name}")
            if value is not None:
                remapped[target_name.format(i=layer_index)] = value

    return remapped


def _lookup_conv_in_channels(
    state_dict: Dict[str, torch.Tensor],
    candidate_keys: Sequence[str],
) -> int:
    for key in candidate_keys:
        value = state_dict.get(key)
        if value is not None and value.ndim >= 2:
            return int(value.shape[1])
    raise KeyError(f"Could not find any of the expected keys: {candidate_keys}")


def infer_pretrained_feature_channels(
    pretrained_mask2former,
) -> Optional[Dict[str, int]]:
    try:
        decoder_state_dict = pretrained_mask2former.model.pixel_level_module.decoder.state_dict()
        return {
            "1": _lookup_conv_in_channels(
                decoder_state_dict,
                ("lateral_convolutions.0.0.weight", "lateral_convolutions.0.weight"),
            ),
            "2": _lookup_conv_in_channels(
                decoder_state_dict,
                ("input_projections.2.0.weight", "input_projections.2.weight"),
            ),
            "3": _lookup_conv_in_channels(
                decoder_state_dict,
                ("input_projections.1.0.weight", "input_projections.1.weight"),
            ),
            "4": _lookup_conv_in_channels(
                decoder_state_dict,
                ("input_projections.0.0.weight", "input_projections.0.weight"),
            ),
        }
    except (AttributeError, KeyError):
        return None


def load_pretrained_mask2former(name_or_path: Optional[str]):
    if not name_or_path:
        return None
    from transformers.models.mask2former.configuration_mask2former import Mask2FormerConfig
    from transformers.models.mask2former.modeling_mask2former import (
        Mask2FormerForUniversalSegmentation,
    )

    config = Mask2FormerConfig.from_pretrained(name_or_path)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.eval()
    return model


def initialize_head_from_pretrained(segmentation_model: nn.Module, pretrained_mask2former) -> None:
    head = segmentation_model.segmentation_model[-1]

    pixel_decoder_state_dict = remap_local_pixel_decoder_state_dict(
        pretrained_mask2former.model.pixel_level_module.decoder.state_dict()
    )
    load_matching_state_dict(head.pixel_decoder, pixel_decoder_state_dict, "GOOSE pixel decoder")

    transformer_state_dict = remap_local_transformer_state_dict(
        pretrained_mask2former.model.transformer_module.state_dict(),
        pretrained_mask2former.class_predictor.state_dict(),
    )
    load_matching_state_dict(head.predictor, transformer_state_dict, "GOOSE transformer decoder")
