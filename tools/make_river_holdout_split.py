#!/usr/bin/env python3
"""Build a river-domain holdout dataset without modifying or copying the source.

Default layout:
  train: 01.한강, 02.낙동강, 03.금강, 04.영산강 (original train preserved)
  val:   01.한강, 02.낙동강, 03.금강, 04.영산강 (original val preserved)
  test:  05.새만금 from the original train + val + test splits

The output consists of symbolic links, so the source dataset remains untouched and
the new layout consumes little additional disk space.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


KINDS = ("images", "labels", "labels_color")
TRAIN_RIVERS = ("01.한강", "02.낙동강", "03.금강", "04.영산강")
HOLDOUT_RIVER = "05.새만금"
SOURCE_SPLITS = ("train", "val", "test")


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def count_files(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="data/goose",
        help="existing dataset root",
    )
    parser.add_argument(
        "--output",
        default="data/goose_holdout_saemangeum",
        help="new river-holdout dataset root (must not already exist)",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().absolute()
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {source}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            f"Output already exists: {output}\n"
            "Choose a new --output path to avoid overwriting existing data."
        )

    summary: dict[str, object] = {
        "source": str(source),
        "output": str(output),
        "train_rivers": list(TRAIN_RIVERS),
        "holdout_river": HOLDOUT_RIVER,
        "test_source_splits": list(SOURCE_SPLITS),
        "counts": {},
    }

    try:
        # Preserve the existing train/val partitions for the four training rivers.
        for kind in KINDS:
            for split in ("train", "val"):
                for river in TRAIN_RIVERS:
                    src = source / kind / split / river
                    if not src.is_dir():
                        raise FileNotFoundError(f"Required directory missing: {src}")
                    link(src, output / kind / split / river)

        # Merge every existing Saemangeum split into one domain-holdout test split.
        for kind in KINDS:
            names_seen: dict[str, Path] = {}
            test_dir = output / kind / "test" / HOLDOUT_RIVER
            test_dir.mkdir(parents=True, exist_ok=True)
            for split in SOURCE_SPLITS:
                src_dir = source / kind / split / HOLDOUT_RIVER
                if not src_dir.is_dir():
                    raise FileNotFoundError(f"Required directory missing: {src_dir}")
                for src in sorted(src_dir.iterdir()):
                    if not src.is_file():
                        continue
                    if src.name in names_seen:
                        raise RuntimeError(
                            f"Filename collision while merging holdout split: {src.name}\n"
                            f"  first: {names_seen[src.name]}\n  next:  {src}"
                        )
                    names_seen[src.name] = src
                    link(src, test_dir / src.name)

        counts: dict[str, dict[str, dict[str, int]]] = {}
        for kind in KINDS:
            counts[kind] = {}
            for split in ("train", "val", "test"):
                counts[kind][split] = {}
                split_dir = output / kind / split
                for river_dir in sorted(split_dir.iterdir()):
                    counts[kind][split][river_dir.name] = count_files(river_dir)
        summary["counts"] = counts

        with (output / "river_holdout_manifest.json").open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

    except Exception:
        # Only remove the directory tree created by this failed invocation. It contains
        # links and metadata, never source files.
        for root, dirs, files in os.walk(output, topdown=False, followlinks=False):
            root_path = Path(root)
            for name in files:
                (root_path / name).unlink()
            for name in dirs:
                path = root_path / name
                if path.is_symlink():
                    path.unlink()
                else:
                    path.rmdir()
        if output.exists():
            output.rmdir()
        raise

    image_counts = counts["images"]
    train_total = sum(image_counts["train"].values())
    val_total = sum(image_counts["val"].values())
    test_total = sum(image_counts["test"].values())
    print(f"Created: {output}")
    print(f"train: {train_total:,} images ({', '.join(TRAIN_RIVERS)})")
    print(f"val:   {val_total:,} images ({', '.join(TRAIN_RIVERS)})")
    print(f"test:  {test_total:,} images ({HOLDOUT_RIVER}, merged train+val+test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
