#!/usr/bin/env python3
"""Benchmark segmentation inference cost for teacher or student models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_student_kd import (
    build_student,
    count_parameters,
    forward_student,
    load_teacher,
    mask2former_outputs_to_semantic_probs,
    model_size_mb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["teacher", "student"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--student_model", choices=["segformer_b0", "segformer_b1"], default="segformer_b1")
    parser.add_argument("--student_local_files_only", action="store_true")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--tta", choices=["none", "hflip"], default="none")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Input scales to average for TTA. Combine with --tta hflip for scale+hflip TTA.",
    )
    parser.add_argument("--output_json", default=None)

    # Teacher reconstruction args. Defaults match this repo's teacher.
    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--teacher_model_factory", default="goose")
    parser.add_argument("--teacher_strict_load", action="store_true")
    parser.add_argument("--backbone_loader", choices=["auto", "native", "hf"], default="auto")
    parser.add_argument(
        "--hf_dinov3_model_name_or_path",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument("--allow_hf_download", dest="hf_local_files_only", action="store_false")
    parser.set_defaults(hf_local_files_only=True)
    parser.add_argument("--dinov3_weights", default=None)
    parser.add_argument(
        "--mask2former_pretrained_model_name_or_path",
        default="facebook/mask2former-swin-large-ade-semantic",
    )
    parser.add_argument("--disable_mask2former_pretrained", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--enable_cls_aux", action="store_true")
    parser.add_argument("--cls_aux_num_classes", type=int, default=0)
    return parser.parse_args()


def load_student_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=True)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.model_type == "teacher":
        teacher_checkpoint = args.teacher_checkpoint or args.checkpoint
        if not teacher_checkpoint:
            raise ValueError("--checkpoint or --teacher_checkpoint is required for teacher benchmark.")
        args.teacher_checkpoint = teacher_checkpoint
        return load_teacher(args, device)

    model = build_student(
        args.student_model,
        args.num_classes,
        local_files_only=args.student_local_files_only,
    )
    if args.checkpoint:
        load_student_checkpoint(model, args.checkpoint)
    model.to(device)
    model.eval()
    return model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_once(
    model: nn.Module,
    model_type: str,
    inputs: torch.Tensor,
    num_classes: int,
    target_size: Tuple[int, int],
) -> torch.Tensor:
    if model_type == "teacher":
        outputs = model(inputs)
        return mask2former_outputs_to_semantic_probs(
            outputs,
            num_classes=num_classes,
            target_size=target_size,
            temperature=1.0,
        )
    return forward_student(model, inputs, target_size)


@torch.no_grad()
def run_once_with_tta(
    model: nn.Module,
    model_type: str,
    inputs: torch.Tensor,
    num_classes: int,
    target_size: Tuple[int, int],
    tta: str,
    scales: List[float],
) -> torch.Tensor:
    if tta == "none" and scales == [1.0]:
        return run_once(model, model_type, inputs, num_classes, target_size)
    if tta not in {"none", "hflip"}:
        raise ValueError(f"Unsupported TTA: {tta}")

    outputs = []
    for scale in scales:
        if scale <= 0:
            raise ValueError(f"Scale must be positive. Got: {scale}")
        if scale == 1.0:
            scaled_inputs = inputs
        else:
            scaled_size = (
                max(1, int(round(inputs.shape[-2] * scale))),
                max(1, int(round(inputs.shape[-1] * scale))),
            )
            scaled_inputs = F.interpolate(
                inputs,
                size=scaled_size,
                mode="bilinear",
                align_corners=False,
            )
        outputs.append(run_once(model, model_type, scaled_inputs, num_classes, target_size))
        if tta == "hflip":
            flipped_inputs = torch.flip(scaled_inputs, dims=[-1])
            flipped_output = run_once(model, model_type, flipped_inputs, num_classes, target_size)
            outputs.append(torch.flip(flipped_output, dims=[-1]))
    return torch.stack(outputs, dim=0).mean(dim=0)


def benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    inputs = torch.randn(args.batch_size, 3, args.height, args.width, device=device)

    amp_dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[args.dtype]
    use_amp = device.type == "cuda" and args.dtype != "fp32"

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            output = run_once(
                model,
                args.model_type,
                inputs,
                args.num_classes,
                (args.height, args.width),
            ) if args.tta == "none" else run_once_with_tta(
                model,
                args.model_type,
                inputs,
                args.num_classes,
                (args.height, args.width),
                args.tta,
                args.scales,
            )
    synchronize(device)

    peak_memory_mb = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                run_once_with_tta(
                    model,
                    args.model_type,
                    inputs,
                    args.num_classes,
                    (args.height, args.width),
                    args.tta,
                    args.scales,
                )
    synchronize(device)

    start = time.perf_counter()
    for _ in range(args.iters):
        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                run_once_with_tta(
                    model,
                    args.model_type,
                    inputs,
                    args.num_classes,
                    (args.height, args.width),
                    args.tta,
                    args.scales,
                )
    synchronize(device)
    elapsed = time.perf_counter() - start

    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    latency_ms = elapsed * 1000.0 / max(1, args.iters)
    throughput = args.batch_size * args.iters / max(elapsed, 1e-12)
    metrics = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint or args.teacher_checkpoint,
        "student_model": args.student_model if args.model_type == "student" else None,
        "num_classes": args.num_classes,
        "input_shape": [args.batch_size, 3, args.height, args.width],
        "output_shape": list(output.shape),
        "dtype": args.dtype,
        "tta": args.tta,
        "scales": args.scales,
        "tta_forwards_per_image": len(args.scales) * (2 if args.tta == "hflip" else 1),
        "device": str(device),
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_ms_per_batch": latency_ms,
        "latency_ms_per_image": latency_ms / args.batch_size,
        "throughput_images_per_sec": throughput,
        "parameters": count_parameters(model),
        "model_size_mb": model_size_mb(model),
        "peak_memory_allocated_mb": peak_memory_mb,
    }
    return metrics


def main() -> None:
    args = parse_args()
    metrics = benchmark(args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
