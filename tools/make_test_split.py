#!/usr/bin/env python3
"""
Carve a held-out `test` split out of `val`, stratified per scene (river) so each
river keeps its proportion. Moves the matching image, label, and color-label.

  images/val/<scene>/<stem>.png            -> images/test/<scene>/
  labels/val/<scene>/<stem>_labelids.png   -> labels/test/<scene>/
  labels_color/val/<scene>/<stem>_color.png-> labels_color/test/<scene>/

Deterministic (fixed seed). Default test fraction = 0.30 of each scene's val set.

Usage:
    python tools/make_test_split.py --root /home/miplab1/dummdumm/goose_data \
        --frac 0.30 --seed 42 [--dry-run]
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

SUBDIRS = {
    "images": ".png",
    "labels": "_labelids.png",
    "labels_color": "_color.png",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/goose")
    ap.add_argument("--frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    img_val_root = root / "images" / "val"
    scenes = sorted(p.name for p in img_val_root.iterdir() if p.is_dir())
    if not scenes:
        print(f"No scenes under {img_val_root}")
        return 1

    rng = random.Random(args.seed)
    grand = {"val_before": 0, "test": 0, "moved_files": 0, "missing": 0}
    print(f"test 분리: 각 강 val의 {args.frac*100:.0f}%  (seed={args.seed})"
          + ("  [DRY-RUN]" if args.dry_run else "") + "\n")

    for scene in scenes:
        stems = sorted(p.stem for p in (img_val_root / scene).glob("*.png"))
        n = len(stems)
        k = int(round(n * args.frac))
        rng.shuffle(stems)
        pick = stems[:k]
        grand["val_before"] += n
        grand["test"] += len(pick)

        for stem in pick:
            for sub, suffix in SUBDIRS.items():
                src = root / sub / "val" / scene / f"{stem}{suffix}"
                dst = root / sub / "test" / scene / f"{stem}{suffix}"
                if not src.exists():
                    grand["missing"] += 1
                    continue
                if args.dry_run:
                    grand["moved_files"] += 1
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                grand["moved_files"] += 1

        print(f"  {scene:<10} val {n:>5} -> test {len(pick):>4}  (남는 val {n-len(pick):>5})")

    print(f"\n=== 요약 ===")
    print(f"  분리 전 val 총: {grand['val_before']:,}")
    print(f"  test 로 이동   : {grand['test']:,}  (이미지 기준)")
    print(f"  남는 val       : {grand['val_before']-grand['test']:,}")
    print(f"  실제 이동 파일 : {grand['moved_files']:,} (이미지+라벨+컬러 3종)")
    if grand["missing"]:
        print(f"  누락(대응파일 없음): {grand['missing']}")
    if args.dry_run:
        print("\n[DRY-RUN] 실제 이동 안 함. --dry-run 빼고 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
