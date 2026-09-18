#!/usr/bin/env python3
"""Convert task-grouped COCO polygons in Labeling_Data to semantic PNG masks."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F


CLASS_NAMES = (
    "land", "bridge", "other", "nps", "algae0", "algae1",
    "algae2", "algae3", "algae4", "turbid", "ambiguous", "nps_algae",
)
TRAIN_CLASS_NAMES = tuple(name for name in CLASS_NAMES if name != "ambiguous")

# Grouping by task prevents adjacent frames from leaking across splits.  The
# selected validation/test tasks jointly cover both sensors and every annotated
# class (category 11/ambiguous has no annotation anywhere in this delivery).
VAL_TASKS = {"Altum/task22", "P1/task31", "P1/task53"}
TEST_TASKS = {"Altum/task24", "P1/task8", "P1/task48", "P1/task52"}


def split_for(group: str) -> str:
    if group in VAL_TASKS:
        return "val"
    if group in TEST_TASKS:
        return "test"
    return "train"


def chunk_splits(group: str, image_count: int, chunk_size: int, seed: int) -> dict[int, str]:
    """Assign contiguous chunks within each task to deterministic 70/15/15 splits."""
    chunk_count = math.ceil(image_count / chunk_size)
    chunk_ids = list(range(chunk_count))
    chunk_ids.sort(
        key=lambda idx: hashlib.sha256(f"{seed}:{group}:{idx}".encode()).digest()
    )
    if chunk_count >= 3:
        val_count = max(1, round(chunk_count * 0.15))
        test_count = max(1, round(chunk_count * 0.15))
    else:
        val_count, test_count = (0, 1) if chunk_count == 2 else (0, 0)
    assignments = {idx: "train" for idx in chunk_ids}
    for idx in chunk_ids[:val_count]:
        assignments[idx] = "val"
    for idx in chunk_ids[val_count : val_count + test_count]:
        assignments[idx] = "test"
    return assignments


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="Labeling_Data")
    parser.add_argument("--output", default="data/labeling_semseg_12cls")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--split-mode", choices=("task", "chunk"), default="task")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--drop-ambiguous", action="store_true")
    parser.add_argument(
        "--multispectral", action="store_true",
        help="Preserve six Altum sensor bands as uint16 CHW .npy files.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}; use --overwrite")
        shutil.rmtree(output)

    records = []
    split_counts = Counter()
    split_classes: dict[str, Counter] = defaultdict(Counter)
    try:
        for sensor in ("Altum", "P1"):
            annotation_dir = source / sensor / "Annotations"
            for annotation_file in sorted(annotation_dir.glob("instances_*.json")):
                task = annotation_file.stem.removeprefix("instances_")
                group = f"{sensor}/{task}"
                coco = json.loads(annotation_file.read_text(encoding="utf-8"))
                categories = {int(c["id"]): c["name"] for c in coco["categories"]}
                expected = {i + 1: name for i, name in enumerate(CLASS_NAMES)}
                if categories != expected:
                    raise ValueError(f"Unexpected categories in {annotation_file}: {categories}")

                annotations = defaultdict(list)
                for ann in coco["annotations"]:
                    annotations[int(ann["image_id"])].append(ann)

                items = sorted(coco["images"], key=lambda x: x["file_name"])
                assignments = chunk_splits(group, len(items), args.chunk_size, args.seed)
                for item_index, item in enumerate(items):
                    split = (
                        split_for(group)
                        if args.split_mode == "task"
                        else assignments[item_index // args.chunk_size]
                    )
                    image_id = int(item["id"])
                    image_path = source / sensor / "Images" / task / item["file_name"]
                    if not image_path.is_file():
                        raise FileNotFoundError(image_path)
                    # Prefixing avoids collisions between sensors/tasks.
                    stem = f"{sensor}_{task}_{image_path.stem}"
                    scene = f"{sensor}_{task}"
                    image_suffix = ".npy" if args.multispectral else ".png"
                    image_out = output / "images" / split / scene / f"{stem}{image_suffix}"
                    label_out = output / "labels" / split / scene / f"{stem}_labelids.png"
                    image_out.parent.mkdir(parents=True, exist_ok=True)
                    label_out.parent.mkdir(parents=True, exist_ok=True)

                    with Image.open(image_path) as raw:
                        source_width, source_height = raw.size
                        annotation_width = int(item["width"])
                        annotation_height = int(item["height"])
                        image_aspect = source_width / source_height
                        annotation_aspect = annotation_width / annotation_height
                        if abs(image_aspect - annotation_aspect) > 1e-3:
                            raise ValueError(f"COCO/image aspect-ratio mismatch: {image_path}")
                        if args.multispectral:
                            if sensor == "Altum":
                                pages = []
                                for page_index in range(6):
                                    raw.seek(page_index)
                                    pages.append(np.array(raw, dtype=np.uint16, copy=True))
                                chw = np.stack(pages, axis=0)
                            else:
                                rgb = np.array(raw.convert("RGB"), dtype=np.uint16, copy=True) * 257
                                chw = np.zeros((6, source_height, source_width), dtype=np.uint16)
                                # Shared layout: Blue, Green, Red, NIR, RedEdge, Thermal.
                                chw[0], chw[1], chw[2] = rgb[..., 2], rgb[..., 1], rgb[..., 0]
                            tensor = torch.from_numpy(chw.astype(np.float32, copy=False))[None]
                            tensor = F.interpolate(
                                tensor, size=(args.height, args.width), mode="bilinear",
                                align_corners=False,
                            )[0].round().clamp_(0, 65535).to(torch.uint16)
                            np.save(image_out, tensor.numpy(), allow_pickle=False)
                        else:
                            raw.convert("RGB").resize(
                                (args.width, args.height), Image.Resampling.BILINEAR
                            ).save(image_out, optimize=True)

                    mask = Image.new("L", (args.width, args.height), 255)
                    draw = ImageDraw.Draw(mask)
                    # Some P1 JPEGs are stored at 8192x5460 while their COCO
                    # polygons use a 4096x2730 coordinate system.
                    sx, sy = args.width / annotation_width, args.height / annotation_height
                    present = set()
                    # Larger regions first lets smaller object polygons retain labels.
                    for ann in sorted(annotations[image_id], key=lambda x: x.get("area", 0), reverse=True):
                        category_id = int(ann["category_id"])
                        if args.drop_ambiguous and category_id == 11:
                            continue
                        # Collapse the removed category-11 slot so nps_algae
                        # becomes class 10 in the observed 11-class setup.
                        class_id = category_id - 1 - int(args.drop_ambiguous and category_id > 11)
                        for polygon in ann.get("segmentation", []):
                            if len(polygon) < 6:
                                continue
                            points = [(polygon[i] * sx, polygon[i + 1] * sy) for i in range(0, len(polygon), 2)]
                            draw.polygon(points, fill=class_id)
                        present.add(class_id)
                        split_classes[split][class_id] += 1
                    mask.save(label_out, optimize=True)
                    split_counts[split] += 1
                    records.append({
                        "sensor": sensor, "task": task, "split": split,
                        "source_image": str(image_path.relative_to(source)),
                        "image": str(image_out.relative_to(output)),
                        "label": str(label_out.relative_to(output)),
                        "classes_present": sorted(present),
                    })

        manifest = {
            "source": str(source),
            "size": [args.width, args.height],
            "ignore_index": 255,
            "input_format": "uint16_chw_npy" if args.multispectral else "rgb_png",
            "input_channels": 6 if args.multispectral else 3,
            "band_order": (
                ["blue", "green", "red", "nir", "red_edge", "thermal"]
                if args.multispectral else ["red", "green", "blue"]
            ),
            "class_names": list(TRAIN_CLASS_NAMES if args.drop_ambiguous else CLASS_NAMES),
            "category_mapping": {
                str(category_id): category_id - 1 - int(args.drop_ambiguous and category_id > 11)
                for category_id in range(1, 13)
                if not (args.drop_ambiguous and category_id == 11)
            },
            "split_policy": (
                "task-grouped deterministic split"
                if args.split_mode == "task"
                else f"within-task contiguous chunks of {args.chunk_size}, deterministic 70/15/15"
            ),
            "validation_tasks": sorted(VAL_TASKS),
            "test_tasks": sorted(TEST_TASKS),
            "split_counts": dict(split_counts),
            "annotation_counts_by_class": {
                split: {
                    (TRAIN_CLASS_NAMES if args.drop_ambiguous else CLASS_NAMES)[k]: v
                    for k, v in sorted(counts.items())
                }
                for split, counts in split_classes.items()
            },
            "records": records,
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise

    print(f"Created: {output}")
    for split in ("train", "val", "test"):
        print(f"{split}: {split_counts[split]} images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
