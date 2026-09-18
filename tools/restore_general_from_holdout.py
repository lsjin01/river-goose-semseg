#!/usr/bin/env python3
"""Create and restore a portable general split from a Saemangeum holdout dataset.

The holdout layout contains the original train/val samples and all Saemangeum
samples, but it omits non-Saemangeum samples from the original test split.  The
``pack`` command records every original sample and copies only those missing
test pairs into a small supplement bundle.  The ``restore`` command combines a
holdout dataset with that bundle without duplicating files by default.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")
HOLDOUT_RIVER = "05.새만금"
SENSOR_SUFFIXES = (
    "_windshield_vis",
    "_front",
    "_camera_left",
    "_camera_right",
    "_realsense",
)


def label_name(image: Path) -> str:
    stem = image.stem
    for suffix in SENSOR_SUFFIXES:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return f"{stem}_labelids.png"


def relative_text(path: Path) -> str:
    return path.as_posix()


def holdout_locations(split: str, river: str, image_name: str, label: str):
    if river == HOLDOUT_RIVER:
        split = "test"
    elif split == "test":
        return None
    return (
        Path("images") / split / river / image_name,
        Path("labels") / split / river / label,
    )


def pack(args: argparse.Namespace) -> int:
    general = Path(args.general).expanduser().resolve()
    holdout = Path(args.holdout).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().absolute()
    if not general.is_dir() or not holdout.is_dir():
        raise FileNotFoundError("Both --general and --holdout must be dataset directories")
    if bundle.exists():
        raise FileExistsError(f"Bundle already exists: {bundle}")

    records = []
    missing = 0
    for split in SPLITS:
        for image in sorted((general / "images" / split).glob("*/*.png")):
            river = image.parent.name
            label = general / "labels" / split / river / label_name(image)
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image}: {label}")
            original_image = image.relative_to(general)
            original_label = label.relative_to(general)
            locations = holdout_locations(split, river, image.name, label.name)
            source = "holdout"
            source_image = source_label = None
            if locations is not None:
                candidate_image, candidate_label = locations
                if (holdout / candidate_image).is_file() and (holdout / candidate_label).is_file():
                    source_image, source_label = candidate_image, candidate_label
                else:
                    locations = None
            if locations is None:
                source = "supplement"
                source_image = Path("supplement") / original_image
                source_label = Path("supplement") / original_label
                missing += 1
            records.append(
                {
                    "image": relative_text(original_image),
                    "label": relative_text(original_label),
                    "source": source,
                    "source_image": relative_text(source_image),
                    "source_label": relative_text(source_label),
                }
            )

    bundle.mkdir(parents=True)
    try:
        for record in records:
            if record["source"] != "supplement":
                continue
            for key, original_key in (("source_image", "image"), ("source_label", "label")):
                destination = bundle / record[key]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(general / record[original_key], destination)
        manifest = {
            "format_version": 1,
            "description": "Restore the original general train/val/test split from holdout + supplement",
            "holdout_river": HOLDOUT_RIVER,
            "sample_count": len(records),
            "supplement_sample_count": missing,
            "records": records,
        }
        (bundle / "general_split_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(bundle)
        raise

    print(f"Created bundle: {bundle}")
    print(f"General samples: {len(records):,}")
    print(f"Already present in holdout: {len(records) - missing:,}")
    print(f"Supplement copied: {missing:,} image/label pairs")
    return 0


def materialize(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def restore(args: argparse.Namespace) -> int:
    holdout = Path(args.holdout).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    manifest_path = bundle / "general_split_manifest.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError(f"Unsupported manifest format: {manifest.get('format_version')}")

    output.mkdir(parents=True)
    try:
        for record in manifest["records"]:
            root = holdout if record["source"] == "holdout" else bundle
            for source_key, destination_key in (("source_image", "image"), ("source_label", "label")):
                source = root / record[source_key]
                if not source.is_file():
                    raise FileNotFoundError(f"Required source is missing: {source}")
                materialize(source, output / record[destination_key], args.mode)
        summary = {
            "source_holdout": str(holdout),
            "source_bundle": str(bundle),
            "mode": args.mode,
            "sample_count": manifest["sample_count"],
        }
        (output / "restored_general_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        shutil.rmtree(output)
        raise
    print(f"Restored general dataset: {output}")
    print(f"Samples: {manifest['sample_count']:,} image/label pairs")
    print(f"Mode: {args.mode}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pack_parser = subparsers.add_parser("pack", help="create manifest + missing-file supplement")
    pack_parser.add_argument("--general", required=True)
    pack_parser.add_argument("--holdout", required=True)
    pack_parser.add_argument("--bundle", required=True)
    pack_parser.set_defaults(func=pack)

    restore_parser = subparsers.add_parser("restore", help="restore the original general split")
    restore_parser.add_argument("--holdout", required=True)
    restore_parser.add_argument("--bundle", required=True)
    restore_parser.add_argument("--output", required=True)
    restore_parser.add_argument("--mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    restore_parser.set_defaults(func=restore)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
