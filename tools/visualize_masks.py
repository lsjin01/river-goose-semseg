#!/usr/bin/env python3
"""
Sanity-check the converted goose_data: for N random samples, render a side-by-side
[ original | colored mask | overlay ] panel so polygon labels can be eyeballed.

Class colors (CATEGORY_1 / 대분류):
    0 background          -> black
    1 농업계 (agri)        -> green
    2 축산계 (livestock)   -> red
    3 하천수면/수변 (water)-> blue

Usage:
    python tools/visualize_masks.py --root /home/miplab1/dummdumm/goose_data \
        --split train --n 12 --out /home/miplab1/dummdumm/viz [--seed 0]
"""
from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import numpy as np
from PIL import Image

# id -> (R,G,B); index 0 is background
PALETTE = [
    (0, 0, 0),        # 0 background
    (0, 200, 0),      # 1 농업계
    (220, 0, 0),      # 2 축산계
    (0, 90, 255),     # 3 하천수면 및 수변
]
CLASS_NAMES = ["background", "농업계", "축산계", "하천수면/수변"]


def colorize(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cid, color in enumerate(PALETTE):
        rgb[mask == cid] = color
    return rgb


def make_panel(image_path: Path, label_path: Path, alpha: float) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(label_path).convert("L"))
    color = colorize(mask)
    color_img = Image.fromarray(color)

    img_arr = np.array(img).astype(np.float32)
    # only blend where there is a foreground label, keep background as-is
    fg = (mask > 0)[..., None]
    blend = np.where(fg, (1 - alpha) * img_arr + alpha * color.astype(np.float32), img_arr)
    overlay = Image.fromarray(blend.clip(0, 255).astype(np.uint8))

    w, h = img.size
    panel = Image.new("RGB", (w * 3 + 8, h), (255, 255, 255))
    panel.paste(img, (0, 0))
    panel.paste(color_img, (w + 4, 0))
    panel.paste(overlay, (w * 2 + 8, 0))
    return panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/goose")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="outputs/visualizations/masks")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefer-multiclass", action="store_true",
                    help="bias sampling toward masks containing >=2 foreground classes")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = sorted(glob.glob(str(root / "labels" / args.split / "*" / "*_labelids.png")))
    if not labels:
        print(f"No labels found under {root}/labels/{args.split}")
        return 1

    random.seed(args.seed)
    if args.prefer_multiclass:
        pool = random.sample(labels, min(len(labels), args.n * 25))
        scored = []
        for lp in pool:
            u = np.unique(np.array(Image.open(lp).convert("L")))
            scored.append((len(set(u.tolist()) - {0}), lp))
        scored.sort(reverse=True)
        picks = [lp for _, lp in scored[:args.n]]
    else:
        picks = random.sample(labels, min(len(labels), args.n))

    print(f"색상 범례: " + ", ".join(f"{i}:{n}" for i, n in enumerate(CLASS_NAMES)))
    print(f"패널 구성: [ 원본 | 컬러마스크 | 오버레이 ]  → {out}\n")
    for i, lp in enumerate(picks):
        label_path = Path(lp)
        stem = label_path.name.replace("_labelids.png", "")
        scene = label_path.parent.name
        image_path = root / "images" / args.split / scene / f"{stem}.png"
        if not image_path.exists():
            print(f"  [skip] missing image for {stem}")
            continue
        mask = np.array(Image.open(label_path).convert("L"))
        present = sorted(set(np.unique(mask).tolist()) - {0})
        names = ", ".join(CLASS_NAMES[c] for c in present) or "(빈 마스크)"
        panel = make_panel(image_path, label_path, args.alpha)
        out_path = out / f"{i:02d}_{scene}_{stem}.png"
        panel.save(out_path)
        print(f"  {out_path.name:<55} 클래스: {names}")

    print(f"\n완료: {out} 에 패널 이미지 저장됨")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
