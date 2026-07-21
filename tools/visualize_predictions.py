#!/usr/bin/env python3
"""
Run the trained model on TEST samples and save [ image | ground-truth | prediction ]
panels for visual inspection.

Reuses the exact model build + preprocessing + Mask2Former semantic inference from
the training repo, so predictions match what the trainer computes.

Class colors (CATEGORY_1 / 대분류):
    0 background -> black, 1 농업계 -> green, 2 축산계 -> red, 3 하천수면/수변 -> blue

Usage (run inside the repo dir so `goose_semseg` is importable):
    cd goose-semseg-icra2026
    CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/goose_env/bin/python \
        ../tools/visualize_predictions.py \
        --run_dir /home/miplab1/dummdumm/outputs/river_pollution_cat1_4cls \
        --ckpt best_epoch_013_miou_0.9411.pt \
        --n 12 --out /home/miplab1/dummdumm/viz_pred --prefer-multiclass
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

# Make the training repo (and its vendored third_party/dinov3) importable,
# mirroring goose-semseg-icra2026/train.py.
_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from goose_semseg.data.dataset import IMAGENET_MEAN, IMAGENET_STD, find_goose_samples
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import BackboneLayersSet, build_segmentation_decoder
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.pretrained.mask2former import (
    infer_pretrained_feature_channels,
    load_pretrained_mask2former,
)

COARSE_PALETTE = np.array([
    (0,   0,   0),    # 0 background
    (0,   180, 0),    # 1 농업계
    (230, 70,  40),   # 2 축산계
    (0,   110, 255),  # 3 하천수면/수변
], dtype=np.uint8)
COARSE_CLASS_NAMES = ["background", "농업계", "축산계", "하천수면/수변"]

FINE_PALETTE = np.array([
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
FINE_CLASS_NAMES = [
    "background",
    "밭·논", "잔재물", "배수로", "비닐하우스", "과수원",
    "축사", "야적퇴비·분뇨", "목장", "분뇨처리시설",
    "부유쓰레기", "연못",
]

PALETTE = COARSE_PALETTE
CLASS_NAMES = COARSE_CLASS_NAMES


def colorize(mask: np.ndarray) -> np.ndarray:
    return PALETTE[np.clip(mask, 0, len(PALETTE) - 1)]


def build_model(args: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    backbone = load_dinov3_backbone(args)
    feature_channels = None
    if not args.disable_mask2former_pretrained:
        pretrained = load_pretrained_mask2former(args.mask2former_pretrained_model_name_or_path)
        feature_channels = infer_pretrained_feature_channels(pretrained)
        args.hidden_dim = int(pretrained.config.hidden_dim)
    model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type="m2f",
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        autocast_dtype=torch.bfloat16,
        freeze_backbone=args.freeze_backbone,
        feature_channels=feature_channels,
        cls_aux_num_classes=(
            int(args.cls_aux_num_classes)
            if getattr(args, "enable_cls_aux", False)
            else 0
        ),
    ).to(device)
    return model


def preprocess(image_path: Path, size):
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img, copy=True)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t, arr  # normalized tensor (for model), uint8 RGB (for display)


def _load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def make_panel(disp_rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> Image.Image:
    h, w = disp_rgb.shape[:2]
    gap = 6
    header = 34
    legend_cols = 4 if len(CLASS_NAMES) <= 4 else 6
    legend_rows = (len(CLASS_NAMES) + legend_cols - 1) // legend_cols
    legend = 8 + legend_rows * 27
    titles = ["INPUT (원본)", "GROUND TRUTH (정답)", "PREDICTION (예측)"]
    cols = [Image.fromarray(disp_rgb), Image.fromarray(colorize(gt)), Image.fromarray(colorize(pred))]

    W = w * 3 + gap * 2
    panel = Image.new("RGB", (W, header + h + legend), (255, 255, 255))
    draw = ImageDraw.Draw(panel)
    font = _load_font(20)
    font_s = _load_font(16)

    for i, (title, im) in enumerate(zip(titles, cols)):
        x = i * (w + gap)
        panel.paste(im, (x, header))
        tw = draw.textlength(title, font=font)
        draw.text((x + (w - tw) / 2, 7), title, fill=(0, 0, 0), font=font)

    # Color legend along the bottom. Fine-class runs use two rows of six.
    legend_ids = list(range(1, len(CLASS_NAMES))) + [0]
    cell_w = W // legend_cols
    for j, cid in enumerate(legend_ids):
        row, col = divmod(j, legend_cols)
        lx = col * cell_w + 8
        ly = header + h + 6 + row * 27
        name = "배경" if cid == 0 else CLASS_NAMES[cid]
        draw.rectangle([lx, ly, lx + 18, ly + 18], fill=tuple(int(v) for v in PALETTE[cid]),
                       outline=(0, 0, 0))
        draw.text((lx + 23, ly + 1), name, fill=(0, 0, 0), font=font_s)
    return panel


def select_samples(samples, n: int, seed: int, prefer_multiclass: bool,
                   required_classes: set[int], cover_all_classes: bool = False,
                   focus_classes: set[int] | None = None):
    """Select deterministically after inspecting GT class presence over the full split."""
    rng = random.Random(seed)
    candidates = []
    for img_p, lbl_p in samples:
        gt = np.asarray(Image.open(lbl_p).convert("L"))
        present = set(np.unique(gt).tolist()) - {0, 255}
        if required_classes and not required_classes.issubset(present):
            continue
        # Prefer more foreground classes, then samples where the rarest present class
        # occupies a meaningful area instead of only a few pixels.
        ratios = [float((gt == c).mean()) for c in present]
        balance = min(ratios) if ratios else 0.0
        focus_ratio = (
            float(np.isin(gt, list(focus_classes)).mean())
            if focus_classes else 0.0
        )
        candidates.append((len(present), balance, rng.random(), frozenset(present),
                           focus_ratio, (img_p, lbl_p)))

    if focus_classes:
        candidates = [row for row in candidates if row[4] > 0]
        candidates.sort(key=lambda x: (x[4], x[0], x[2]), reverse=True)
    elif required_classes:
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    elif prefer_multiclass:
        candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    else:
        rng.shuffle(candidates)

    if cover_all_classes and not required_classes:
        class_freq = {
            c: sum(c in row[3] for row in candidates)
            for c in range(1, max((max(row[3], default=0) for row in candidates), default=0) + 1)
        }
        uncovered = {c for c, freq in class_freq.items() if freq > 0}
        selected = []
        selected_ids = set()
        while uncovered and len(selected) < n:
            best_idx = None
            best_score = -1.0
            for idx, row in enumerate(candidates):
                if idx in selected_ids:
                    continue
                gain = row[3] & uncovered
                score = sum(1.0 / class_freq[c] for c in gain)
                if score > best_score:
                    best_idx, best_score = idx, score
            if best_idx is None or best_score <= 0:
                break
            selected.append(candidates[best_idx])
            selected_ids.add(best_idx)
            uncovered -= candidates[best_idx][3]
        selected.extend(
            row for idx, row in enumerate(candidates)
            if idx not in selected_ids and len(selected) < n
        )
        chosen = selected[:n]
    else:
        chosen = candidates[:n]

    return [row[5] for row in chosen], len(candidates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="outputs/river_pollution_cat1_4cls")
    ap.add_argument("--ckpt", default=None, help="checkpoint filename in run_dir (default: highest best_*.pt)")
    ap.add_argument("--data_path", default=None, help="override data_path (default: from train_args.json)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="outputs/visualizations/predictions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefer-multiclass", action="store_true")
    ap.add_argument(
        "--cover-all-classes",
        action="store_true",
        help="select samples so every class present in the split appears at least once",
    )
    ap.add_argument(
        "--focus-classes",
        type=int,
        nargs="+",
        default=None,
        help="select samples with the highest combined GT pixel ratio for these class IDs",
    )
    ap.add_argument(
        "--stems-file",
        default=None,
        help="optional text file with one image stem per line; restrict selection to these samples",
    )
    ap.add_argument(
        "--require-all-classes",
        action="store_true",
        help="only visualize samples whose GT contains every foreground class (1, 2, 3)",
    )
    cli = ap.parse_args()

    run_dir = Path(cli.run_dir)
    with (run_dir / "train_args.json").open() as fp:
        args = SimpleNamespace(**json.load(fp))
    if cli.data_path:
        args.data_path = cli.data_path

    global PALETTE, CLASS_NAMES
    if int(args.num_classes) == 4:
        PALETTE = COARSE_PALETTE
        CLASS_NAMES = COARSE_CLASS_NAMES
    elif int(args.num_classes) == 12:
        PALETTE = FINE_PALETTE
        CLASS_NAMES = FINE_CLASS_NAMES
    else:
        raise ValueError(
            f"Visualization palette is defined for 4 or 12 classes, got {args.num_classes}."
        )

    ckpt_path = run_dir / cli.ckpt if cli.ckpt else max(
        run_dir.glob("best_*.pt"), key=lambda p: p.stat().st_mtime
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"checkpoint: {ckpt_path.name}")
    print(f"data_path : {args.data_path}  split: {cli.split}")

    model = build_model(args, device)
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
    print(f"loaded (epoch {state.get('epoch')}, best_miou {state.get('best_val_miou')}); "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    size = (int(args.resize_width), int(args.resize_height))
    samples = find_goose_samples(Path(args.data_path), cli.split)
    if cli.stems_file:
        stems = {
            line.strip()
            for line in Path(cli.stems_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        samples = [sample for sample in samples if sample[0].stem in stems]
        missing_stems = stems - {sample[0].stem for sample in samples}
        if missing_stems:
            raise FileNotFoundError(
                f"{len(missing_stems)} requested stems were not found in {cli.split}: "
                + ", ".join(sorted(missing_stems))
            )
    required_classes = set(range(1, int(args.num_classes))) if cli.require_all_classes else set()
    picks, eligible = select_samples(
        samples, cli.n, cli.seed, cli.prefer_multiclass, required_classes,
        cli.cover_all_classes, set(cli.focus_classes or []),
    )
    if cli.require_all_classes:
        print(f"GT에 전경 3클래스가 모두 있는 샘플: {eligible}/{len(samples)}")
        if eligible < cli.n:
            print(f"요청한 {cli.n}개보다 적어 가능한 {eligible}개만 저장합니다.")

    out = Path(cli.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n패널 [ 원본 | GT | 예측 ]  클래스 수: {len(CLASS_NAMES)} (하단 색상 범례 참조)\n")

    inter_iou = np.zeros(len(CLASS_NAMES))
    union = np.zeros(len(CLASS_NAMES))
    for i, (img_p, lbl_p) in enumerate(picks):
        x, disp = preprocess(img_p, size)
        gt = np.array(Image.open(lbl_p).convert("L").resize(size, Image.NEAREST))
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), amp:
            outputs = model(x.unsqueeze(0).to(device))
            scores = mask2former_semantic_scores(outputs, target_size=gt.shape[-2:])
        pred = scores.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        for c in range(len(CLASS_NAMES)):
            inter_iou[c] += np.logical_and(pred == c, gt == c).sum()
            union[c] += np.logical_or(pred == c, gt == c).sum()

        acc = (pred == gt).mean() * 100
        present = sorted(set(np.unique(gt).tolist()) - {0})
        names = ", ".join(CLASS_NAMES[c] for c in present) or "(배경뿐)"
        panel = make_panel(disp, gt, pred)
        stem = img_p.stem
        panel.save(out / f"{i:02d}_{img_p.parent.name}_{stem}.png")
        print(f"  {i:02d} {stem:<40} px-acc {acc:5.1f}%  GT클래스: {names}")

    iou = np.where(union > 0, inter_iou / np.maximum(union, 1), np.nan)
    print("\n=== 이 샘플들의 클래스별 IoU ===")
    for c, n in enumerate(CLASS_NAMES):
        print(f"  {n}: {iou[c]:.3f}" if union[c] > 0 else f"  {n}: (해당 없음)")
    print(f"\n완료: {out} 에 {len(picks)}개 패널 저장")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
