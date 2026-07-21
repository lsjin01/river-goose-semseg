"""Checkpoint loading and TTA inference for DINOv3 + Mask2Former.

Slim, single-backend port of the original goosetools.checkpoint_inference module.
Only supports the DINOv3 + Mask2Former path; convnext / vit branches were dropped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from PIL import Image
from torchvision import transforms

# --- sys.path setup so this script works when run directly or imported ---
_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _p in (_TOOLS, _REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from goose_semseg.models.builder import (  # noqa: E402
    BackboneLayersSet,
    build_segmentation_decoder,
)
from goose_semseg.models.backbone.loader import load_dinov3_backbone  # noqa: E402
from goose_semseg.pretrained.mask2former import (  # noqa: E402
    infer_pretrained_feature_channels,
    load_pretrained_mask2former,
)
from goose_semseg.utils.checkpoint import (  # noqa: E402
    load_model_state_allowing_token_specialization,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LoadedCheckpointModel:
    checkpoint_args: dict
    crop: bool
    device: torch.device
    model: object
    num_classes: int
    payload: dict
    resize_size: Tuple[int, int]


def strip_file_prefix(path: Path | str) -> Path:
    return Path(str(path).removeprefix("file://"))


def load_checkpoint_payload(checkpoint_path: Path) -> dict:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(
            "Expected a Mask2Former training checkpoint with a model_state_dict field."
        )
    return payload


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedCheckpointModel:
    """Load a DINOv3+Mask2Former training checkpoint into an evaluation bundle."""
    payload = load_checkpoint_payload(checkpoint_path)

    checkpoint_args = dict(payload.get("args", {}))
    checkpoint_args.setdefault("backbone_loader", "hf")
    checkpoint_args.setdefault(
        "hf_dinov3_model_name_or_path",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    checkpoint_args.setdefault("hf_local_files_only", True)
    checkpoint_args.setdefault("dinov3_weights", None)
    checkpoint_args.setdefault(
        "mask2former_pretrained_model_name_or_path",
        "facebook/mask2former-swin-large-ade-semantic",
    )
    checkpoint_args.setdefault("disable_mask2former_pretrained", False)
    checkpoint_args.setdefault("freeze_backbone", False)
    checkpoint_args.setdefault("hidden_dim", 256)
    checkpoint_args.setdefault("num_classes", 12)
    checkpoint_args.setdefault("enable_cls_aux", False)
    checkpoint_args.setdefault("cls_aux_num_classes", 3)
    checkpoint_args["device"] = str(device)
    args = argparse.Namespace(**checkpoint_args)

    backbone = load_dinov3_backbone(args)

    feature_channels = None
    if not args.disable_mask2former_pretrained:
        pretrained_mask2former = load_pretrained_mask2former(
            args.mask2former_pretrained_model_name_or_path
        )
        feature_channels = infer_pretrained_feature_channels(pretrained_mask2former)
        args.hidden_dim = int(pretrained_mask2former.config.hidden_dim)

    model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type="m2f",
        hidden_dim=args.hidden_dim,
        num_classes=int(args.num_classes),
        autocast_dtype=torch.bfloat16,
        freeze_backbone=bool(args.freeze_backbone),
        feature_channels=feature_channels,
        cls_aux_num_classes=(
            int(args.cls_aux_num_classes) if bool(args.enable_cls_aux) else 0
        ),
    ).to(device)

    missing_keys, unexpected_keys, _expanded_keys = (
        load_model_state_allowing_token_specialization(
            model,
            payload["model_state_dict"],
        )
    )
    if missing_keys:
        print(
            "Warning: checkpoint load left {} model tensors initialized from "
            "the current model. First keys: {}".format(
                len(missing_keys), ", ".join(missing_keys[:5])
            )
        )
    if unexpected_keys:
        print(
            "Warning: checkpoint had {} unused tensors. First keys: {}".format(
                len(unexpected_keys), ", ".join(unexpected_keys[:5])
            )
        )

    model.eval()

    resize_width = checkpoint_args.get("resize_width")
    resize_height = checkpoint_args.get("resize_height")
    if resize_width is None or resize_height is None:
        raise ValueError(
            "Checkpoint args must include resize_width and resize_height."
        )

    return LoadedCheckpointModel(
        checkpoint_args=checkpoint_args,
        crop=False,
        device=device,
        model=model,
        num_classes=int(args.num_classes),
        payload=payload,
        resize_size=(int(resize_width), int(resize_height)),
    )


def outputs_to_semantic_logits(
    outputs,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    """Generic HF-Mask2Former-style outputs -> semantic logits.

    Kept for compatibility with callers that may pass HF outputs objects;
    for the goose_semseg model use ``run_model_logits`` which knows how to
    dispatch via ``model.predict``.
    """
    class_logits = outputs.class_queries_logits[..., :-1]
    mask_logits = outputs.masks_queries_logits

    class_probs = torch.softmax(class_logits, dim=-1)
    mask_probs = torch.sigmoid(mask_logits)
    semantic_logits = torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)
    if semantic_logits.shape[-2:] != target_size:
        semantic_logits = F.interpolate(
            semantic_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
    return semantic_logits


def run_model_logits(
    bundle: LoadedCheckpointModel,
    pixel_values: torch.Tensor,
) -> torch.Tensor:
    outputs = bundle.model.predict(
        pixel_values,
        rescale_to=pixel_values.shape[-2:],
    )
    class_probs = torch.softmax(outputs["pred_logits"].float(), dim=-1)[..., :-1]
    mask_probs = torch.sigmoid(outputs["pred_masks"].float())
    return torch.einsum("bqc,bqhw->bchw", class_probs, mask_probs)


def round_to_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / float(multiple))) * multiple)


def aspect_ratio_scaled_size(
    original_size: Tuple[int, int],
    short_side: int,
    multiple: int,
) -> Tuple[int, int]:
    original_width, original_height = original_size
    scale = float(short_side) / float(min(original_width, original_height))
    return (
        round_to_multiple(original_width * scale, multiple),
        round_to_multiple(original_height * scale, multiple),
    )


def run_sliding_window_logits(
    bundle: LoadedCheckpointModel,
    pixel_values: torch.Tensor,
    crop_size: int,
    stride: int,
) -> torch.Tensor:
    _, _, image_height, image_width = pixel_values.shape
    crop_height = min(int(crop_size), max(int(crop_size), image_height))
    crop_width = min(int(crop_size), max(int(crop_size), image_width))
    padded_height = max(image_height, crop_height)
    padded_width = max(image_width, crop_width)

    pad_bottom = padded_height - image_height
    pad_right = padded_width - image_width
    if pad_bottom or pad_right:
        pixel_values = F.pad(pixel_values, (0, pad_right, 0, pad_bottom), value=0.0)

    h_grids = max(padded_height - crop_height + stride - 1, 0) // stride + 1
    w_grids = max(padded_width - crop_width + stride - 1, 0) // stride + 1
    logits_sum: Optional[torch.Tensor] = None
    count_map = pixel_values.new_zeros((1, 1, padded_height, padded_width))

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop_height, padded_height)
            x2 = min(x1 + crop_width, padded_width)
            y1 = max(y2 - crop_height, 0)
            x1 = max(x2 - crop_width, 0)

            crop = pixel_values[:, :, y1:y2, x1:x2]
            crop_logits = run_model_logits(bundle, crop).to(torch.float32)
            if logits_sum is None:
                logits_sum = crop_logits.new_zeros(
                    (1, crop_logits.shape[1], padded_height, padded_width)
                )
            logits_sum[:, :, y1:y2, x1:x2] += crop_logits
            count_map[:, :, y1:y2, x1:x2] += 1.0

    if logits_sum is None:
        raise RuntimeError("No sliding-window crops were produced.")
    logits = logits_sum / count_map.clamp_min(1.0)
    return logits[:, :, :image_height, :image_width]


def run_tta_forward(
    bundle: LoadedCheckpointModel,
    pixel_values: torch.Tensor,
    sliding_window: bool,
    crop_size: int,
    stride: int,
) -> torch.Tensor:
    if sliding_window:
        return run_sliding_window_logits(
            bundle,
            pixel_values,
            crop_size=crop_size,
            stride=stride,
        )
    return run_model_logits(bundle, pixel_values).to(torch.float32)


def export_predictions_for_paths(
    image_paths: Sequence[Path],
    checkpoint_path: Path,
    output_dir: Path,
    device_name: Optional[str] = None,
    output_name_resolver: Optional[Callable[[Path], str]] = None,
    multi_scales: Sequence[float] = (1.0,),
    horizontal_flip: bool = False,
    sliding_window: bool = False,
    crop_size: int = 1024,
    stride: int = 672,
) -> Path:
    device = torch.device(
        device_name or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    bundle = build_model_from_checkpoint(checkpoint_path, device)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_name_resolver is None:
        output_name_resolver = lambda path: path.name

    if not multi_scales:
        raise ValueError("multi_scales must include at least one value.")
    scales = tuple(float(scale) for scale in multi_scales)
    if any(scale <= 0 for scale in scales):
        raise ValueError("All multi_scales values must be positive.")
    if crop_size <= 0:
        raise ValueError("crop_size must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    to_tensor = transforms.ToTensor()
    imagenet_normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    print(f"device={bundle.device}")
    print(f"checkpoint={checkpoint_path}")
    print(f"backend=dinov3_mask2former")
    print(f"crop={bundle.crop}")
    print(f"resize_size={bundle.resize_size}")
    print(f"multi_scales={scales}")
    print(f"horizontal_flip={horizontal_flip}")
    print(f"sliding_window={sliding_window}")
    print(f"crop_size={crop_size}")
    print(f"stride={stride}")
    print(f"prediction_image_count={len(image_paths)}")
    print(f"generated_predictions_dir={output_dir}")

    with torch.no_grad():
        for image_path in tqdm.tqdm(image_paths, desc="Generating predictions"):
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            target_hw = (original_size[1], original_size[0])

            semantic_logits_sum: Optional[torch.Tensor] = None
            for scale in scales:
                if sliding_window:
                    short_side = round_to_multiple(min(bundle.resize_size) * scale, 32)
                    scaled_size = aspect_ratio_scaled_size(
                        original_size,
                        short_side=short_side,
                        multiple=32,
                    )
                else:
                    # The DINOv3 adapter fuses stride-8/16/32 features, so
                    # TTA sizes must stay aligned to the stride-32 grid.
                    scaled_size = (
                        round_to_multiple(bundle.resize_size[0] * scale, 32),
                        round_to_multiple(bundle.resize_size[1] * scale, 32),
                    )
                model_image = image.resize(scaled_size, Image.BILINEAR)
                pixel_values = imagenet_normalize(to_tensor(model_image))

                pixel_values = pixel_values.unsqueeze(0).to(bundle.device)
                scale_logits = run_tta_forward(
                    bundle,
                    pixel_values,
                    sliding_window=sliding_window,
                    crop_size=crop_size,
                    stride=stride,
                )
                if horizontal_flip:
                    flipped_pixel_values = torch.flip(pixel_values, dims=[-1])
                    flipped_logits = run_tta_forward(
                        bundle,
                        flipped_pixel_values,
                        sliding_window=sliding_window,
                        crop_size=crop_size,
                        stride=stride,
                    )
                    flipped_logits = torch.flip(flipped_logits, dims=[-1])
                    scale_logits = (scale_logits + flipped_logits) / 2.0
                if scale_logits.shape[-2:] != target_hw:
                    scale_logits = F.interpolate(
                        scale_logits,
                        size=target_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                if semantic_logits_sum is None:
                    semantic_logits_sum = scale_logits
                else:
                    semantic_logits_sum += scale_logits

            if semantic_logits_sum is None:
                raise RuntimeError("No logits were produced during multi-scale inference.")
            semantic_logits = semantic_logits_sum / float(len(scales))
            destination_path = output_dir / output_name_resolver(image_path)

            prediction = (
                semantic_logits.argmax(dim=1)
                .squeeze(0)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )
            Image.fromarray(np.asarray(prediction, dtype=np.uint8)).save(destination_path)

    return output_dir
