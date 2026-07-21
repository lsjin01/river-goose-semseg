#!/usr/bin/env python3
"""
Make HUMAN-VIEWABLE colored copies of the goose_data label masks.

WHY a separate folder:
  The real training labels in  goose_data/labels/  MUST stay as raw class-id
  (mode "L", pixel value = class index 0..3) because the loader does
  Image.open(...).convert("L") and uses the value as the class. Coloring them in
  place (even via a palette PNG) corrupts the indices -> training breaks.

  So we write RGB copies to  goose_data/labels_color/  (ignored by the loader).

Color map (CATEGORY_1 / 대분류):
    0 background          -> black
    1 농업계 (agri)        -> green
    2 축산계 (livestock)   -> red
    3 하천수면/수변 (water)-> blue

Usage:
    python tools/colorize_labels.py --root /home/miplab1/dummdumm/goose_data \
        [--out <dir>]   # default: <root>/labels_color
        [--workers 10] [--limit 0]
"""
from __future__ import annotations

import argparse
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

PALETTE = np.array([
    (0,   0,   0),    # 0  background
    (240, 220, 0),    # 1  밭, 논              yellow
    (204, 80, 160),   # 2  잔재물              magenta
    (0,   158, 115),  # 3  배수로              teal
    (145, 70, 255),   # 4  비닐하우스          violet
    (110, 210, 0),    # 5  과수원              lime
    (230, 120, 0),    # 6  축사                orange
    (135, 70,  20),   # 7  야적퇴비 및 가축분뇨 brown
    (255, 105, 180),  # 8  목장                pink
    (210, 40,  40),   # 9  분뇨개별 처리시설   red
    (0,   90,  220),  # 10 부유쓰레기          blue
    (0,   200, 230),  # 11 연못                cyan
], dtype=np.uint8)


def _colorize_one(args: tuple) -> bool:
    src, dst = args
    mask = np.array(Image.open(src).convert("L"))
    mask = np.clip(mask, 0, len(PALETTE) - 1)
    rgb = PALETTE[mask]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, "RGB").save(dst)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/goose")
    ap.add_argument("--out", default=None, help="default: <root>/labels_color")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out) if args.out else root / "labels_color"
    labels = sorted(glob.glob(str(root / "labels" / "*" / "*" / "*_labelids.png")))
    if args.limit:
        labels = labels[:args.limit]
    if not labels:
        print(f"No labels under {root}/labels/")
        return 1

    tasks = []
    for src in labels:
        rel = Path(src).relative_to(root / "labels")          # train/<scene>/<stem>_labelids.png
        dst = out_root / rel.with_name(rel.name.replace("_labelids.png", "_color.png"))
        tasks.append((src, str(dst)))

    print(f"컬러 라벨 생성: {len(tasks):,}개  ->  {out_root}")
    print("색상: 0=검정(배경) 1=초록(농업계) 2=빨강(축산계) 3=파랑(수변)\n")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_colorize_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            done += 1
            if i % 5000 == 0:
                print(f"  ... {i:,}/{len(tasks):,}")
    print(f"\n완료: {done:,}개 컬러 라벨 -> {out_root}")
    print("※ 학습용 라벨(goose_data/labels)은 그대로 유지됨 — 이 폴더는 검수 전용")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
