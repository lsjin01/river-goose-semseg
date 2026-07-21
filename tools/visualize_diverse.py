"""Diverse, class-coverage-driven segmentation visualization for the baseline checkpoint.

Picks val samples so that every 중분류 class (1..11) appears at least once, forces the
two low-performing classes 2(잔재물) & 10(부유쓰레기) to be present, emphasises rare/small
classes (7,8,9,10,11), prefers multi-class scenes, excludes background-only frames, and
keeps a mix of success and failure cases. Saves image/GT/pred/overlay grids with class
names + per-image IoU, a class legend, and a selection CSV.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Patch

# --- sys.path setup ---
_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _p in (_TOOLS, _REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

if not hasattr(torch.amp, "GradScaler"):
    torch.amp.GradScaler = torch.cuda.amp.GradScaler

from inference import build_model_from_checkpoint, run_model_logits  # noqa: E402
from goose_semseg.data.dataset import GooseSegmentationDataset  # noqa: E402
from goose_semseg.data.coarse_labels import GOOSE_FINE_CLASS_NAMES  # noqa: E402

# ----- Korean font -----
_FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
if Path(_FONT_PATH).exists():
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

DATA_PATH = "data/goose_cat2"
CHECKPOINT = "outputs/teacher/best.pt"
COLORMAP_PATH = _TOOLS / "common" / "goose_colormap.json"
OUTPUT_DIR = Path("outputs/visualizations/diverse")

NUM_CLASSES = 12
IGNORE_INDEX = 255
FG_CLASSES = list(range(1, NUM_CLASSES))         # 1..11 (exclude background 0)
MANDATORY = [2, 10]                              # 잔재물, 부유쓰레기
RARE = [7, 8, 9, 10, 11]                         # rare / small-object classes
SPLIT = "val"
TARGET_N = 18                                    # number of visualizations (12~20)
MIN_FG_RATIO = 0.05                              # exclude background-only frames
MIN_FG_CLASSES = 2                               # prefer multi-class scenes


def load_colormap() -> Dict[int, Tuple[int, int, int]]:
    import json
    with open(COLORMAP_PATH, "r", encoding="utf-8") as fp:
        raw = json.load(fp)
    cmap = {}
    for k, v in raw.items():
        cmap[int(k)] = tuple(int(c) if c > 1 else int(c * 255) for c in v)
    return cmap


def mask_to_color(mask: np.ndarray, cmap: Dict[int, Tuple]) -> np.ndarray:
    h, w = mask.shape
    out = np.full((h, w, 3), 128, dtype=np.uint8)  # gray = ignore / unmapped
    for cid, color in cmap.items():
        region = mask == cid
        if region.any():
            out[region] = color
    return out


def overlay(image: np.ndarray, mask: np.ndarray, cmap, alpha=0.5) -> np.ndarray:
    cm = mask_to_color(mask, cmap).astype(np.float32)
    return np.clip(image.astype(np.float32) * (1 - alpha) + cm * alpha, 0, 255).astype(np.uint8)


def scan_gt(dataset: GooseSegmentationDataset, cache: Path) -> List[dict]:
    """Per-image GT class presence + pixel counts (cached)."""
    if cache.exists():
        rows = []
        with open(cache, newline="", encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                r["index"] = int(r["index"])
                r["fg_ratio"] = float(r["fg_ratio"])
                r["counts"] = {int(k): int(v) for k, v in
                               (p.split(":") for p in r["counts"].split("|") if p)}
                rows.append(r)
        print(f"Loaded GT scan cache: {cache} ({len(rows)} rows)")
        return rows

    rows = []
    for idx in tqdm(range(len(dataset)), desc="Scanning GT masks"):
        img_path, lbl_path = dataset.samples[idx]
        lbl = np.array(Image.open(lbl_path))
        vals, cnts = np.unique(lbl, return_counts=True)
        counts = {int(v): int(c) for v, c in zip(vals, cnts)
                  if v != IGNORE_INDEX and 0 <= v < NUM_CLASSES}
        total = int(lbl.size)
        fg = sum(c for k, c in counts.items() if k != 0)
        rows.append({
            "index": idx,
            "image_path": str(img_path),
            "counts": counts,
            "fg_ratio": fg / total if total else 0.0,
        })
    with open(cache, "w", newline="", encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["index", "image_path", "fg_ratio", "counts"])
        for r in rows:
            w.writerow([r["index"], r["image_path"], f"{r['fg_ratio']:.6f}",
                        "|".join(f"{k}:{v}" for k, v in r["counts"].items())])
    print(f"Wrote GT scan cache: {cache}")
    return rows


def select_samples(rows: List[dict]) -> List[int]:
    """Coverage/rarity greedy selection."""
    # class frequency over images (foreground classes only)
    freq = {c: 0 for c in FG_CLASSES}
    for r in rows:
        for c in r["counts"]:
            if c in freq and r["counts"][c] > 0:
                freq[c] += 1
    rarity = {c: 1.0 / np.sqrt(max(freq[c], 1)) for c in FG_CLASSES}
    print("Class image-frequency (val):")
    for c in FG_CLASSES:
        print(f"  {c:2d} {GOOSE_FINE_CLASS_NAMES[c]:<14} freq={freq[c]}")

    def fg_classes(r):
        return [c for c, n in r["counts"].items() if c != 0 and n > 0]

    # eligible: enough foreground + multi-class (relax later if needed)
    eligible = [r for r in rows
                if r["fg_ratio"] >= MIN_FG_RATIO and len(fg_classes(r)) >= MIN_FG_CLASSES]
    if not eligible:
        eligible = [r for r in rows if r["fg_ratio"] >= MIN_FG_RATIO]
    by_index = {r["index"]: r for r in rows}

    selected: List[int] = []
    covered = set()

    def add(idx):
        if idx not in selected:
            selected.append(idx)
            covered.update(fg_classes(by_index[idx]))

    # 1) force mandatory classes with the richest (most multi-class) scene
    for mc in MANDATORY:
        cands = [r for r in eligible if r["counts"].get(mc, 0) > 0]
        if cands:
            cands.sort(key=lambda r: (len(fg_classes(r)), r["counts"][mc], r["fg_ratio"]),
                       reverse=True)
            add(cands[0]["index"])

    # 2) greedy set-cover over all foreground classes, weighted by rarity
    pool = [r for r in eligible if r["index"] not in selected]
    while covered != set(FG_CLASSES):
        remaining = set(FG_CLASSES) - covered
        best, best_score = None, -1.0
        for r in pool:
            if r["index"] in selected:
                continue
            gain = sum(rarity[c] for c in fg_classes(r) if c in remaining)
            if gain <= 0:
                continue
            score = gain + 0.01 * len(fg_classes(r)) + 0.001 * r["fg_ratio"]
            if score > best_score:
                best, best_score = r, score
        if best is None:
            break
        add(best["index"])

    # 3) enrich rare classes: ensure >=2 selected images per rare class if available
    for rc in RARE:
        have = sum(1 for i in selected if by_index[i]["counts"].get(rc, 0) > 0)
        cands = sorted([r for r in eligible
                        if r["counts"].get(rc, 0) > 0 and r["index"] not in selected],
                       key=lambda r: (len(fg_classes(r)), r["counts"][rc], r["fg_ratio"]),
                       reverse=True)
        for r in cands:
            if have >= 2:
                break
            add(r["index"])
            have += 1

    # 4) top up to TARGET_N with the most class-rich scenes
    if len(selected) < TARGET_N:
        rich = sorted([r for r in eligible if r["index"] not in selected],
                      key=lambda r: (len(fg_classes(r)), r["fg_ratio"]), reverse=True)
        for r in rich:
            if len(selected) >= TARGET_N:
                break
            add(r["index"])

    return selected[:max(TARGET_N, len(set()))]


@torch.no_grad()
def predict(bundle, image_tensor, device, target_hw) -> np.ndarray:
    pv = image_tensor.unsqueeze(0).to(device)
    logits = run_model_logits(bundle, pv)
    if logits.shape[-2:] != target_hw:
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
    return logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int32)


def per_class_iou(gt: np.ndarray, pred: np.ndarray) -> Dict[int, float]:
    """IoU for classes present in GT or pred (excluding ignore)."""
    valid = gt != IGNORE_INDEX
    g, p = gt[valid], pred[valid]
    ious = {}
    classes = set(np.unique(g).tolist()) | set(np.unique(p).tolist())
    for c in sorted(classes):
        if c < 0 or c >= NUM_CLASSES:
            continue
        inter = int(((g == c) & (p == c)).sum())
        union = int(((g == c) | (p == c)).sum())
        if union > 0:
            ious[int(c)] = inter / union
    return ious


def names(ids) -> str:
    return ", ".join(f"{i}:{GOOSE_FINE_CLASS_NAMES.get(i, '?')}" for i in ids)


def save_legend(cmap, out: Path):
    handles = [Patch(facecolor=np.array(cmap[c]) / 255.0, edgecolor="black",
                     label=f"{c}: {GOOSE_FINE_CLASS_NAMES[c]}") for c in range(NUM_CLASSES)]
    fig = plt.figure(figsize=(4, 5))
    fig.legend(handles=handles, loc="center", frameon=True, fontsize=12,
               title="중분류 12-class palette")
    plt.axis("off")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def visualize(image_np, gt, pred, cmap, gt_ious, img_miou, overall_miou,
              title, case_type, save_path):
    gt_present = sorted([c for c in np.unique(gt) if c != IGNORE_INDEX and 0 <= c < NUM_CLASSES])
    pred_present = sorted([c for c in np.unique(pred) if 0 <= c < NUM_CLASSES])

    gt_color = mask_to_color(gt, cmap)
    pred_color = mask_to_color(pred, cmap)
    pred_ov = overlay(image_np, pred, cmap, 0.5)
    diff = (gt != pred) & (gt != IGNORE_INDEX)
    err = image_np.copy()
    err[diff] = [255, 0, 0]

    fig, ax = plt.subplots(2, 3, figsize=(19, 11))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    panels = [
        (ax[0, 0], image_np, "Original Image"),
        (ax[0, 1], gt_color, "GT Mask"),
        (ax[0, 2], overlay(image_np, gt, cmap, 0.5), "GT Overlay"),
        (ax[1, 0], pred_color, "Prediction Mask"),
        (ax[1, 1], pred_ov, "Prediction Overlay"),
        (ax[1, 2], err, f"Error Map (red=wrong, {diff.mean()*100:.1f}%)"),
    ]
    for a, img, t in panels:
        a.imshow(img); a.set_title(t, fontsize=12); a.axis("off")

    gt_iou_str = "  ".join(f"{c}:{GOOSE_FINE_CLASS_NAMES[c]}={gt_ious.get(c, float('nan')):.2f}"
                           for c in gt_present)
    txt = (f"[{case_type}]  per-image mIoU = {img_miou:.4f}   (checkpoint best val mIoU = {overall_miou})\n"
           f"GT classes  : {names(gt_present)}\n"
           f"Pred classes: {names(pred_present)}\n"
           f"per-class IoU (GT∪Pred): {gt_iou_str}")
    fig.text(0.01, 0.005, txt, fontsize=10, va="bottom", ha="left",
             bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="gray"))
    fig.tight_layout(rect=[0, 0.10, 1, 0.97])
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cmap = load_colormap()
    save_legend(cmap, OUTPUT_DIR / "legend.png")
    print(f"Legend saved -> {OUTPUT_DIR/'legend.png'}")

    # dataset at 512x512 (matches trainer val eval)
    dataset = GooseSegmentationDataset(root=DATA_PATH, split=SPLIT,
                                       resize_size=(512, 512), flip_prob=0.0,
                                       enable_random_crop=False)
    print(f"Dataset: {len(dataset)} {SPLIT} samples")

    rows = scan_gt(dataset, OUTPUT_DIR / "val_gt_scan.csv")
    selected = select_samples(rows)
    print(f"\nSelected {len(selected)} candidate indices: {selected}")

    # model
    bundle = build_model_from_checkpoint(Path(CHECKPOINT), device)
    best_miou = bundle.payload.get("best_val_miou", "?")
    if isinstance(best_miou, float):
        best_miou = f"{best_miou:.4f}"
    print(f"Loaded checkpoint (best_val_miou={best_miou})")

    # inference + per-image IoU
    results = []
    for idx in tqdm(selected, desc="Inference"):
        img_t, gt_t = dataset[idx]
        gt = np.array(gt_t)
        pred = predict(bundle, img_t, device, gt.shape)
        ious = per_class_iou(gt, pred)
        gt_present = [c for c in np.unique(gt) if c != IGNORE_INDEX and 0 <= c < NUM_CLASSES]
        # per-image mIoU over GT-present classes
        gt_ious = [ious[c] for c in gt_present if c in ious]
        img_miou = float(np.mean(gt_ious)) if gt_ious else 0.0
        results.append({"index": idx, "gt": gt, "pred": pred, "ious": ious,
                        "gt_present": [int(c) for c in gt_present],
                        "pred_present": [int(c) for c in np.unique(pred) if 0 <= c < NUM_CLASSES],
                        "img_miou": img_miou})

    # case labels: success / failure / mixed by per-image mIoU
    mious = sorted(r["img_miou"] for r in results)
    lo = np.percentile(mious, 33)
    hi = np.percentile(mious, 67)
    for r in results:
        r["case"] = ("failure" if r["img_miou"] <= lo
                     else "success" if r["img_miou"] >= hi else "mixed")

    # render in mIoU order (worst first so failures are easy to find)
    results.sort(key=lambda r: r["img_miou"])
    csv_rows = []
    for rank, r in enumerate(results, 1):
        idx = r["index"]
        img_path, _ = dataset.samples[idx]
        img_t, _ = dataset[idx]
        # de-normalize for display
        from goose_semseg.data.dataset import IMAGENET_MEAN, IMAGENET_STD
        disp = (img_t * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)
        image_np = (disp.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        stem = Path(img_path).stem
        save_path = OUTPUT_DIR / f"viz_{rank:02d}_{r['case']}_miou{r['img_miou']:.3f}_{stem}.png"
        title = f"#{rank}  [{r['case'].upper()}]  {stem}  (idx {idx})"
        visualize(image_np, r["gt"], r["pred"], cmap, r["ious"], r["img_miou"],
                  best_miou, title, r["case"], save_path)

        csv_rows.append({
            "rank": rank,
            "case_type": r["case"],
            "dataset_index": idx,
            "image_path": str(img_path),
            "per_image_miou": f"{r['img_miou']:.4f}",
            "num_gt_classes": len(r["gt_present"]),
            "gt_class_ids": " ".join(map(str, r["gt_present"])),
            "gt_class_names": " | ".join(GOOSE_FINE_CLASS_NAMES[c] for c in r["gt_present"]),
            "pred_class_ids": " ".join(map(str, r["pred_present"])),
            "pred_class_names": " | ".join(GOOSE_FINE_CLASS_NAMES.get(c, "?") for c in r["pred_present"]),
            "per_class_iou": " ".join(f"{c}:{r['ious'][c]:.3f}" for c in sorted(r["ious"])),
            "viz_file": save_path.name,
        })

    # selection CSV
    csv_path = OUTPUT_DIR / "selected_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)

    # coverage report
    covered = set()
    for r in results:
        covered.update(r["gt_present"])
    missing = [c for c in FG_CLASSES if c not in covered]
    print(f"\n=== DONE ===")
    print(f"Visualizations: {len(results)}  -> {OUTPUT_DIR}")
    print(f"Class coverage (GT, fg 1..11): covered={sorted(covered)}  missing={missing}")
    print(f"Mandatory present: " +
          ", ".join(f"{c}({GOOSE_FINE_CLASS_NAMES[c]})={'Y' if c in covered else 'N'}" for c in MANDATORY))
    print(f"Selection CSV: {csv_path}")
    print(f"per-image mIoU range: {min(mious):.3f} .. {max(mious):.3f}")


if __name__ == "__main__":
    main()
