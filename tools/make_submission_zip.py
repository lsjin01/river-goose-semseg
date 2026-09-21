#!/usr/bin/env python3
"""
Create a Codabench submission zip from a DINOv3+Mask2Former training
checkpoint (.pt), end-to-end.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# --- sys.path setup so this script works when run directly ---
_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _p in (_TOOLS, _REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from inference import (  # noqa: E402
    export_predictions_for_paths,
    strip_file_prefix,
)


SCRIPT_DIR = _TOOLS
DEFAULT_COMMON_DIR = _TOOLS / "common"

SENSOR_SUFFIXES: Tuple[str, ...] = (
    "_camera_left",
    "_windshield_vis",
    "_front",
    "_realsense",
)

DEFAULT_LIST_FILES: Tuple[str, ...] = (
    "text file with ALICE scenes.txt",
    "text file with MuCAR-3 scenes.txt",
    "text file with Spotv1 scenes.txt",
    "text file with Spotv2 scenes.txt",
)
EXCLUDED_FINE_CLASS_IDS = {0, 7, 9, 35, 44, 56, 61, 63}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Codabench submission zip directly from a Mask2Former "
            "checkpoint."
        )
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        required=True,
        help="Path to the GOOSE dataset root.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a Mask2Former training checkpoint (.pt).",
    )
    parser.add_argument(
        "--generated_predictions_dir",
        type=Path,
        default=None,
        help=(
            "Directory where generated prediction PNGs will be written before "
            "building the submission zip."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to export when using --checkpoint. Default: test",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference. Default: cuda if available else cpu.",
    )
    parser.add_argument(
        "--multi_scales",
        nargs="+",
        type=float,
        default=[1.0],
        help=(
            "Test-time inference scales. Example: --multi_scales 0.5 1.0 1.5 2.0"
        ),
    )
    parser.add_argument(
        "--horizontal_flip",
        action="store_true",
        help="Use horizontal flip test-time augmentation.",
    )
    parser.add_argument(
        "--sliding_window",
        action="store_true",
        help="Use aspect-ratio-preserving sliding-window inference.",
    )
    parser.add_argument(
        "--crop_size",
        type=int,
        default=1024,
        help="Square crop size for --sliding_window.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=672,
        help="Sliding-window stride for --sliding_window.",
    )
    parser.add_argument(
        "--scene_lists_dir",
        type=Path,
        default=DEFAULT_COMMON_DIR,
        help=(
            "Directory that contains the official txt files listing target scenes. "
            "Default: <repo>/tools/common"
        ),
    )
    parser.add_argument(
        "--filter_to_scene_lists",
        action="store_true",
        help=(
            "Restrict inference/output to the official scene-list subset. "
            "By default the script now predicts every PNG under the requested split."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where the output PNGs will be written.",
    )
    parser.add_argument(
        "--output_zip",
        type=Path,
        required=True,
        help="Path to the final submission zip file.",
    )
    parser.add_argument(
        "--list_files",
        nargs="+",
        default=list(DEFAULT_LIST_FILES),
        help="Txt files to read from --scene_lists_dir.",
    )
    parser.add_argument(
        "--expected_count",
        type=int,
        default=None,
        help="Optional expected number of submission files. Default: target count.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=64,
        help="Number of fine classes used for the prediction class chart.",
    )
    parser.add_argument(
        "--class_chart_path",
        type=Path,
        default=None,
        help=(
            "Optional path for the predicted class distribution chart. "
            "Default: <output_zip_stem>_class_histogram.png"
        ),
    )
    parser.add_argument(
        "--no_class_chart",
        action="store_true",
        help="Do not write the predicted class distribution chart.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    checkpoint_path = strip_file_prefix(args.checkpoint)
    if checkpoint_path.suffix != ".pt":
        raise ValueError(
            "Checkpoint inference currently supports Mask2Former training checkpoints (.pt) only."
        )
    if args.generated_predictions_dir is None:
        args.generated_predictions_dir = args.output_dir.parent / (
            args.output_dir.name + "_generated_predictions"
        )

    if args.generated_predictions_dir.resolve() == args.output_dir.resolve():
        raise ValueError(
            "--generated_predictions_dir must be different from --output_dir."
        )
    if not args.multi_scales:
        raise ValueError("--multi_scales must include at least one value.")
    if any(scale <= 0 for scale in args.multi_scales):
        raise ValueError("--multi_scales values must be positive.")
    if args.crop_size <= 0:
        raise ValueError("--crop_size must be positive.")
    if args.stride <= 0:
        raise ValueError("--stride must be positive.")
    if args.num_classes <= 0:
        raise ValueError("--num_classes must be positive.")


def strip_sensor_suffix(stem: str) -> str:
    for suffix in SENSOR_SUFFIXES:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def strip_labelids_suffix(stem: str) -> str:
    suffix = "_labelids"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def load_target_names(scene_lists_dir: Path, list_files: Sequence[str]) -> List[str]:
    target_names: List[str] = []
    for filename in list_files:
        path = scene_lists_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing scene list file: {path}")
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    target_names.append(line)
    return list(dict.fromkeys(target_names))


def build_name_index(
    paths: Iterable[Path],
) -> Tuple[Dict[str, Path], Dict[Tuple[str, str], Path]]:
    exact_base_map: Dict[str, Path] = {}
    prefix_timestamp_map: Dict[Tuple[str, str], Path] = {}

    for path in paths:
        # Normalize both raw image names (sensor suffix) and generated
        # submission names (_labelids suffix) to the same base identifier.
        base = strip_labelids_suffix(strip_sensor_suffix(path.stem))
        exact_base_map[base] = path

        try:
            prefix, _frame_idx, timestamp = base.rsplit("_", 2)
        except ValueError:
            continue
        prefix_timestamp_map[(prefix, timestamp)] = path

    return exact_base_map, prefix_timestamp_map


def resolve_target_path(
    target_name: str,
    exact_base_map: Dict[str, Path],
    prefix_timestamp_map: Dict[Tuple[str, str], Path],
    missing_message_prefix: str,
) -> Path:
    target_base = strip_labelids_suffix(Path(target_name).stem)
    source = exact_base_map.get(target_base)
    if source is not None:
        return source

    try:
        prefix, _frame_idx, timestamp = target_base.rsplit("_", 2)
    except ValueError as exc:
        raise ValueError(f"Could not parse target filename: {target_name}") from exc

    source = prefix_timestamp_map.get((prefix, timestamp))
    if source is None:
        raise FileNotFoundError(f"{missing_message_prefix}: {target_name}")
    return source


def collect_prediction_files(
    predictions_dir: Path,
) -> List[Path]:
    if not predictions_dir.exists():
        raise FileNotFoundError(
            f"Predictions directory does not exist: {predictions_dir}"
        )

    prediction_files = list(predictions_dir.rglob("*.png"))
    if not prediction_files:
        raise FileNotFoundError(f"No prediction PNGs found under: {predictions_dir}")
    return prediction_files


def collect_image_paths(dataset_root: Path, split: str) -> List[Path]:
    split_root = dataset_root / "images" / split
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory does not exist: {split_root}")

    image_paths = sorted(split_root.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under: {split_root}")
    return image_paths


def select_image_paths_for_targets(
    image_paths: Sequence[Path],
    target_names: Sequence[str],
) -> List[Path]:
    exact_base_map, prefix_timestamp_map = build_name_index(image_paths)
    selected_paths = [
        resolve_target_path(
            target_name=target_name,
            exact_base_map=exact_base_map,
            prefix_timestamp_map=prefix_timestamp_map,
            missing_message_prefix="No test image matched target",
        )
        for target_name in target_names
    ]
    return list(dict.fromkeys(selected_paths))


def prepare_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def copy_submission_files(
    target_names: Sequence[str],
    output_dir: Path,
    exact_base_map: Dict[str, Path],
    prefix_timestamp_map: Dict[Tuple[str, str], Path],
) -> List[Path]:
    created_files: List[Path] = []
    missing_targets: List[str] = []

    for target_name in target_names:
        try:
            source = resolve_target_path(
                target_name=target_name,
                exact_base_map=exact_base_map,
                prefix_timestamp_map=prefix_timestamp_map,
                missing_message_prefix="No prediction matched target",
            )
        except FileNotFoundError:
            missing_targets.append(target_name)
            continue

        destination = output_dir / target_name
        shutil.copy2(source, destination)
        created_files.append(destination)

    if missing_targets:
        preview = ", ".join(missing_targets[:10])
        extra = "" if len(missing_targets) <= 10 else f" ... (+{len(missing_targets) - 10})"
        raise FileNotFoundError(
            f"Missing predictions for {len(missing_targets)} targets: {preview}{extra}"
        )

    return created_files


def copy_prediction_files(
    prediction_files: Sequence[Path],
    output_dir: Path,
) -> List[Path]:
    created_files: List[Path] = []
    for source in sorted(prediction_files):
        destination = output_dir / source.name
        shutil.copy2(source, destination)
        created_files.append(destination)
    return created_files


def write_submission_zip(output_zip: Path, created_files: Sequence[Path]) -> None:
    if output_zip.exists():
        output_zip.unlink()
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        for path in sorted(created_files):
            zf.write(path, arcname=path.name)


def validate_submission_zip(output_zip: Path, expected_count: Optional[int]) -> None:
    with zipfile.ZipFile(output_zip) as zf:
        names = [name for name in zf.namelist() if not name.endswith("/")]
        png_names = [name for name in names if name.lower().endswith(".png")]
        has_subdirs = any("/" in name for name in names)

    if has_subdirs:
        raise ValueError(
            "Submission zip contains subdirectories; expected root-level PNGs only."
        )
    if len(names) != len(png_names):
        raise ValueError("Submission zip contains non-PNG files.")
    if expected_count is not None and len(png_names) != expected_count:
        raise ValueError(
            f"Submission zip contains {len(png_names)} PNGs, expected {expected_count}."
        )


def write_prediction_class_chart(
    mask_paths: Sequence[Path],
    chart_path: Path,
    num_classes: int,
) -> None:
    counts = np.zeros(num_classes, dtype=np.int64)
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            mask = np.asarray(image)
        valid = (mask >= 0) & (mask < num_classes)
        if np.any(valid):
            counts += np.bincount(
                mask[valid].astype(np.int64, copy=False).reshape(-1),
                minlength=num_classes,
            )

    total = int(counts.sum())
    ratios = counts.astype(np.float64) / float(total) if total > 0 else counts.astype(np.float64)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Warning: skipped class distribution chart because matplotlib is unavailable: {exc}")
        return

    chart_path.parent.mkdir(parents=True, exist_ok=True)
    x_values = np.arange(num_classes)
    labels = [str(class_id) for class_id in range(num_classes)]
    colors = [
        "#cbd5e1" if class_id in EXCLUDED_FINE_CLASS_IDS else "#3b82f6"
        for class_id in range(num_classes)
    ]

    fig, ax = plt.subplots(figsize=(max(16.0, num_classes * 0.32), 6.0))
    ax.bar(x_values, ratios, color=colors, width=0.82)
    ax.set_title("Submission Prediction Class Distribution")
    ax.set_xlabel("Class ID")
    ax.set_ylabel("Predicted Pixel Ratio")
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_xticks(x_values)
    tick_labels = ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    for tick_label, class_id in zip(tick_labels, range(num_classes)):
        if class_id in EXCLUDED_FINE_CLASS_IDS:
            tick_label.set_color("#dc2626")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)


def build_prediction_output_name(image_path: Path) -> str:
    return strip_sensor_suffix(image_path.stem) + "_labelids.png"


def main() -> None:
    args = parse_args()
    validate_args(args)

    dataset_root = strip_file_prefix(args.dataset_root)
    image_paths = collect_image_paths(dataset_root, args.split)
    target_names: Optional[List[str]] = None
    if args.filter_to_scene_lists:
        target_names = load_target_names(args.scene_lists_dir, args.list_files)
        image_paths = select_image_paths_for_targets(image_paths, target_names)

    expected_count = (
        args.expected_count
        if args.expected_count is not None
        else (len(target_names) if target_names is not None else len(image_paths))
    )

    predictions_dir = export_predictions_for_paths(
        image_paths=image_paths,
        checkpoint_path=strip_file_prefix(args.checkpoint),
        output_dir=args.generated_predictions_dir,
        device_name=args.device,
        output_name_resolver=build_prediction_output_name,
        multi_scales=args.multi_scales,
        horizontal_flip=args.horizontal_flip,
        sliding_window=args.sliding_window,
        crop_size=args.crop_size,
        stride=args.stride,
    )

    prediction_files = collect_prediction_files(
        predictions_dir=predictions_dir,
    )

    prepare_output_dir(args.output_dir)
    if target_names is not None:
        exact_base_map, prefix_timestamp_map = build_name_index(prediction_files)
        created_files = copy_submission_files(
            target_names=target_names,
            output_dir=args.output_dir,
            exact_base_map=exact_base_map,
            prefix_timestamp_map=prefix_timestamp_map,
        )
    else:
        created_files = copy_prediction_files(
            prediction_files=prediction_files,
            output_dir=args.output_dir,
        )
    write_submission_zip(args.output_zip, created_files)
    validate_submission_zip(args.output_zip, expected_count)
    if not args.no_class_chart:
        chart_path = args.class_chart_path
        if chart_path is None:
            chart_path = args.output_zip.with_name(
                f"{args.output_zip.stem}_class_histogram.png"
            )
        write_prediction_class_chart(created_files, chart_path, args.num_classes)

    if target_names is not None:
        print(f"target_count={len(target_names)}")
    print(f"image_count={len(image_paths)}")
    print(f"created_count={len(created_files)}")
    print(f"output_dir={args.output_dir}")
    print(f"output_zip={args.output_zip}")
    if not args.no_class_chart:
        print(f"class_chart={chart_path}")


if __name__ == "__main__":
    main()
