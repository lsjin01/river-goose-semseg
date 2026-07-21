#!/usr/bin/env python3
"""Create a tiny, reproducible GOOSE dataset without modifying the source."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
SENSOR_SUFFIXES = (
    "_windshield_vis",
    "_front",
    "_camera_left",
    "_camera_right",
    "_realsense",
)


def label_name_for(image_path: Path) -> str:
    stem = image_path.stem
    for suffix in SENSOR_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}_labelids.png"


def paired_samples(source: Path, split: str) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for image_path in sorted((source / "images" / split).glob("*/*.png")):
        label_path = source / "labels" / split / image_path.parent.name / label_name_for(image_path)
        if label_path.is_file():
            pairs.append((image_path, label_path))
    if not pairs:
        raise FileNotFoundError(f"No paired samples found for split {split!r} under {source}")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("goose_data_cat2"))
    parser.add_argument("--output", type=Path, default=Path("data/sample_goose"))
    parser.add_argument("--count", type=int, default=1, help="pairs to copy per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)

    rng = random.Random(args.seed)
    manifest: dict[str, object] = {
        "source": str(source),
        "seed": args.seed,
        "count_per_split": args.count,
        "splits": {},
    }
    try:
        for split in SPLITS:
            pairs = paired_samples(source, split)
            selected = rng.sample(pairs, k=min(args.count, len(pairs)))
            records = []
            for image_path, label_path in selected:
                scene = image_path.parent.name
                image_dest = output / "images" / split / scene / image_path.name
                label_dest = output / "labels" / split / scene / label_path.name
                image_dest.parent.mkdir(parents=True, exist_ok=True)
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, image_dest)
                shutil.copy2(label_path, label_dest)
                records.append(
                    {
                        "image": str(image_dest.relative_to(output)),
                        "label": str(label_dest.relative_to(output)),
                    }
                )
            manifest["splits"][split] = records
        with (output / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        raise

    print(f"Created sample dataset at {output}")
    for split in SPLITS:
        print(f"  {split}: {len(manifest['splits'][split])} pair(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
