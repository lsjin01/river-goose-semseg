#!/usr/bin/env python3
"""
Convert the river-pollution polygon-JSON dataset into the directory layout that
goose-semseg-icra2026 expects (semantic segmentation, CATEGORY_2 / 중분류 = 11 classes).

The CLS auxiliary head learns 대분류 (CATEGORY_1, 3 classes); it is derived from
these fine ids at train time via the fine->coarse table in
goose_semseg/data/coarse_labels.py, so the masks only need to carry 중분류 ids.

Output layout (matches goose_semseg/data/dataset.py -> find_goose_samples):

    <out>/
      images/train/<scene>/<stem>.png
      images/val/<scene>/<stem>.png
      labels/train/<scene>/<stem>_labelids.png
      labels/val/<scene>/<stem>_labelids.png

Each label PNG is single-channel (mode "L"); every pixel value is a 중분류 class id:

    0 = background              (no polygon)               [대분류: -]
    1 = 밭, 논                                              [대분류: 농업계]
    2 = 잔재물                                              [대분류: 농업계]
    3 = 배수로                                              [대분류: 농업계]
    4 = 비닐하우스                                          [대분류: 농업계]
    5 = 과수원                                              [대분류: 농업계]
    6 = 축사                                                [대분류: 축산계]
    7 = 야적퇴비 및 가축분뇨                                 [대분류: 축산계]
    8 = 목장 (목축지)                                        [대분류: 축산계]
    9 = 분뇨개별 처리시설                                     [대분류: 축산계]
   10 = 부유쓰레기                                           [대분류: 하천수면 및 수변]
   11 = 연못 (웅덩이, 저수지)                                 [대분류: 하천수면 및 수변]

So num_classes = 12 (0 = background + 11 foreground). Set this in config/default.yaml.

Reads straight from the .zip files under <dataset_root>; nothing is extracted to
disk except the converted PNGs.

Usage:
    python tools/convert_to_goose.py \
        --dataset-root /home/miplab1/dummdumm/dataset \
        --out /home/miplab1/dummdumm/goose_data_cat2 \
        [--limit 50]      # convert only N images per zip (quick smoke test)
        [--workers 8]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

# --- class mapping (CATEGORY_2 / 중분류) -------------------------------------
# 0 is reserved for background; foreground ids start at 1.
# Keys are the exact CATEGORY_2 strings as they appear in the label JSON
# (verified by a full scan of all 96,340 annotations -- mind the commas,
# parentheses and spacing, e.g. "밭, 논", "목장 (목축지)").
CATEGORY2_TO_ID: Dict[str, int] = {
    "밭, 논": 1,
    "잔재물": 2,
    "배수로": 3,
    "비닐하우스": 4,
    "과수원": 5,
    "축사": 6,
    "야적퇴비 및 가축분뇨": 7,
    "목장 (목축지)": 8,
    "분뇨개별 처리시설": 9,
    "부유쓰레기": 10,
    "연못 (웅덩이, 저수지)": 11,
}
BACKGROUND_ID = 0

# Pair image (원천) zips with label (라벨링) zips by the 2-digit river index
# embedded in the filename, e.g. TS_01.한강 <-> TL_01.한강.
SPLIT_DIRS = [
    ("Training", "train"),
    ("Validation", "val"),
]
IMAGE_SUBDIR = "01.원천데이터"
LABEL_SUBDIR = "02.라벨링데이터"
_INDEX_RE = re.compile(r"_(\d{2})\.")


def _river_index(zip_name: str) -> Optional[str]:
    m = _INDEX_RE.search(zip_name)
    return m.group(1) if m else None


def _scene_name(label_zip: Path) -> str:
    # e.g. "TL_01.한강.zip" -> "01.한강"
    stem = label_zip.name
    if stem.lower().endswith(".zip"):
        stem = stem[:-4]
    parts = stem.split("_", 1)
    return parts[1] if len(parts) == 2 else stem


def _build_image_index(image_zip: Path) -> Dict[str, str]:
    """Map  <stem>.tif (basename, lowercased)  ->  internal zip member name."""
    index: Dict[str, str] = {}
    with zipfile.ZipFile(image_zip) as zf:
        for member in zf.namelist():
            if member.lower().endswith(".tif"):
                base = Path(member).name.lower()
                index[base] = member
    return index


def _polygons_to_mask(
    annotations: List[dict],
    width: int,
    height: int,
    unknown_cats: "Counter[str]",
) -> Tuple[Image.Image, int, int]:
    """Rasterise all polygons of one image into an L-mode label-id mask.

    Class id is looked up from CATEGORY_2 (중분류).  Returns (mask, drawn,
    skipped); any CATEGORY_2 string not in CATEGORY2_TO_ID is counted in
    `unknown_cats` so the caller can surface unexpected labels.  Later polygons
    overwrite earlier ones where they overlap (painter's order = JSON order).
    """
    mask = Image.new("L", (width, height), BACKGROUND_ID)
    draw = ImageDraw.Draw(mask)
    drawn = skipped = 0
    for ann in annotations:
        if str(ann.get("DRAWING", "")).lower() != "polygon":
            skipped += 1
            continue
        cat2 = ann.get("CATEGORY_2")
        class_id = CATEGORY2_TO_ID.get(cat2)
        if class_id is None:
            skipped += 1
            unknown_cats[str(cat2)] += 1
            continue
        coords = ann.get("SEGMENTATION") or []
        if len(coords) < 6:  # need >= 3 points
            skipped += 1
            continue
        pts = [(float(coords[i]), float(coords[i + 1])) for i in range(0, len(coords) - 1, 2)]
        draw.polygon(pts, fill=class_id)
        drawn += 1
    return mask, drawn, skipped


def _process_zip_pair(task: dict) -> dict:
    label_zip = Path(task["label_zip"])
    image_zip = Path(task["image_zip"])
    out_root = Path(task["out"])
    split = task["split"]
    scene = task["scene"]
    limit = task["limit"]

    img_out_dir = out_root / "images" / split / scene
    lbl_out_dir = out_root / "labels" / split / scene
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    image_index = _build_image_index(image_zip)

    n_ok = n_missing_img = n_bad_json = n_empty = 0
    unknown_cats: "Counter[str]" = Counter()
    with zipfile.ZipFile(label_zip) as lzf, zipfile.ZipFile(image_zip) as izf:
        json_members = [m for m in lzf.namelist() if m.lower().endswith(".json")]
        if limit:
            json_members = json_members[:limit]
        for member in json_members:
            try:
                rec = json.loads(lzf.read(member).decode("utf-8"))
            except Exception:
                n_bad_json += 1
                continue
            images_meta = rec.get("IMAGES", {})
            file_name = images_meta.get("FILE_NAME")  # e.g. L01_..._00699.tif
            if not file_name:
                n_bad_json += 1
                continue
            stem = Path(file_name).stem
            width = int(images_meta.get("WIDTH") or 512)
            height = int(images_meta.get("HEIGHT") or 512)

            internal = image_index.get(file_name.lower())
            if internal is None:
                n_missing_img += 1
                continue

            # image: tif -> RGB png
            try:
                im = Image.open(io.BytesIO(izf.read(internal))).convert("RGB")
            except Exception:
                n_missing_img += 1
                continue
            if im.size != (width, height):
                width, height = im.size  # trust the actual pixels

            mask, drawn, _ = _polygons_to_mask(
                rec.get("ANNOTATIONS", []), width, height, unknown_cats
            )
            if drawn == 0:
                n_empty += 1  # kept anyway: an all-background sample is still valid

            im.save(img_out_dir / f"{stem}.png")
            mask.save(lbl_out_dir / f"{stem}_labelids.png")
            n_ok += 1

    return {
        "scene": scene,
        "split": split,
        "ok": n_ok,
        "missing_img": n_missing_img,
        "bad_json": n_bad_json,
        "empty_mask": n_empty,
        "unknown_cats": dict(unknown_cats),
    }


def _collect_tasks(dataset_root: Path, out: Path, limit: int) -> List[dict]:
    tasks: List[dict] = []
    for src_dir, split in SPLIT_DIRS:
        label_dir = dataset_root / src_dir / LABEL_SUBDIR
        image_dir = dataset_root / src_dir / IMAGE_SUBDIR
        if not label_dir.is_dir() or not image_dir.is_dir():
            print(f"[warn] skip split {src_dir!r}: missing subdir", file=sys.stderr)
            continue
        # index image zips by river index
        image_by_index = {}
        for z in image_dir.glob("*.zip"):
            idx = _river_index(z.name)
            if idx:
                image_by_index[idx] = z
        for label_zip in sorted(label_dir.glob("*.zip")):
            idx = _river_index(label_zip.name)
            image_zip = image_by_index.get(idx)
            if image_zip is None:
                print(f"[warn] no image zip paired with {label_zip.name}", file=sys.stderr)
                continue
            tasks.append({
                "label_zip": str(label_zip),
                "image_zip": str(image_zip),
                "out": str(out),
                "split": split,
                "scene": _scene_name(label_zip),
                "limit": limit,
            })
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-root", default="data/raw")
    ap.add_argument("--out", default="data/goose_cat2")
    ap.add_argument("--limit", type=int, default=0, help="max images per zip (0 = all)")
    ap.add_argument("--workers", type=int, default=min(8, len(SPLIT_DIRS) * 5))
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    tasks = _collect_tasks(dataset_root, out, args.limit)
    if not tasks:
        print("No (image, label) zip pairs found.", file=sys.stderr)
        return 1

    print(f"Class mapping (CATEGORY_2 / 중분류): {CATEGORY2_TO_ID}  +  background={BACKGROUND_ID}")
    print(f"num_classes to set in config = {len(CATEGORY2_TO_ID) + 1}")
    print(f"Output root: {out}")
    print(f"Converting {len(tasks)} zip pair(s) with {args.workers} worker(s)"
          + (f"  [limit {args.limit}/zip]" if args.limit else "") + " ...\n")

    totals = {"ok": 0, "missing_img": 0, "bad_json": 0, "empty_mask": 0}
    unknown_totals: "Counter[str]" = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_zip_pair, t): t for t in tasks}
        for fut in as_completed(futs):
            r = fut.result()
            for k in totals:
                totals[k] += r[k]
            unknown_totals.update(r.get("unknown_cats", {}))
            print(f"  [{r['split']:>5}] {r['scene']:<12} "
                  f"ok={r['ok']:>6}  missing_img={r['missing_img']:>4}  "
                  f"bad_json={r['bad_json']:>3}  empty_mask={r['empty_mask']:>5}")

    print("\n=== DONE ===")
    print(f"  images written : {totals['ok']}")
    print(f"  missing images : {totals['missing_img']}")
    print(f"  bad json       : {totals['bad_json']}")
    print(f"  empty masks    : {totals['empty_mask']} (all-background, kept)")
    if unknown_totals:
        print(f"  UNKNOWN CATEGORY_2 (skipped, not in mapping):")
        for name, cnt in unknown_totals.most_common():
            print(f"      {cnt:>8}  {name!r}")
    else:
        print(f"  unknown CATEGORY_2 : 0  (every polygon mapped to a 중분류 id)")
    print(f"\nNext: point config/default.yaml -> data_path: {out}")
    print("      and set  num_classes: 12   (also resize_width/height: 512, enable_cls_aux: true)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
