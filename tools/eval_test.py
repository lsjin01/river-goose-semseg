#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on the full TEST split and report per-class (대분류) IoU
plus mIoU. Reuses the repo's model build, preprocessing, Mask2Former semantic inference,
and confusion-matrix metric so numbers match the trainer's val computation.

Usage:
    cd goose-semseg-icra2026
    CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/goose_env/bin/python ../tools/eval_test.py \
        --run_dir /home/miplab1/dummdumm/outputs/river_pollution_cat1_4cls \
        --ckpt best_epoch_013_miou_0.9411.pt --split test --batch_size 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from goose_semseg.data.dataset import GooseSegmentationDataset
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import BackboneLayersSet, build_segmentation_decoder
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.pretrained.mask2former import (
    infer_pretrained_feature_channels,
    load_pretrained_mask2former,
)
from goose_semseg.utils.metrics import compute_mean_iou, update_confusion_matrix

CLASS_NAMES = ["background", "농업계", "축산계", "하천수면/수변"]


def build_model(args, device):
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
        cls_aux_num_classes=0,
    ).to(device)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="outputs/river_pollution_cat1_4cls")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--num_workers", type=int, default=8)
    cli = ap.parse_args()

    run_dir = Path(cli.run_dir)
    args = SimpleNamespace(**json.load((run_dir / "train_args.json").open()))
    if cli.data_path:
        args.data_path = cli.data_path
    ckpt_path = run_dir / cli.ckpt if cli.ckpt else max(
        run_dir.glob("best_*.pt"), key=lambda p: p.stat().st_mtime)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = int(args.num_classes)
    ignore_index = int(args.ignore_index)

    model = build_model(args, device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()
    print(f"ckpt={ckpt_path.name} (epoch {state.get('epoch')}, best_val_miou={state.get('best_val_miou'):.4f})")

    ds = GooseSegmentationDataset(args.data_path, cli.split,
                                  resize_size=(int(args.resize_width), int(args.resize_height)))
    loader = DataLoader(ds, batch_size=cli.batch_size, shuffle=False, drop_last=False,
                        num_workers=cli.num_workers, pin_memory=True)
    print(f"{cli.split} samples: {len(ds)}  batches: {len(loader)}\n")

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"eval {cli.split}"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(images)
                scores = mask2former_semantic_scores(outputs, target_size=labels.shape[-2:])
            preds = scores.argmax(dim=1).cpu()
            update_confusion_matrix(cm, preds, labels, num_classes, ignore_index)

    m = cm.float()
    inter = torch.diag(m)
    union = m.sum(1) + m.sum(0) - inter
    iou = torch.where(union > 0, inter / union.clamp_min(1), torch.full_like(inter, float("nan")))
    gt_px = m.sum(1)

    print(f"\n=== {cli.split} 셋 대분류별 IoU ({len(ds)} images) ===")
    print(f"{'ID':>2} {'class':<16} {'IoU':>8} {'GT px share':>12}")
    total_px = float(gt_px.sum())
    for c in range(num_classes):
        share = 100.0 * float(gt_px[c]) / total_px if total_px > 0 else 0.0
        iou_s = f"{iou[c].item():.4f}" if union[c] > 0 else "  n/a "
        print(f"{c:>2} {CLASS_NAMES[c]:<16} {iou_s:>8} {share:>11.2f}%")

    miou_all = compute_mean_iou(cm)
    fg = iou[1:num_classes]
    miou_fg = float(fg[~torch.isnan(fg)].mean())
    print(f"\nmIoU (배경 포함, 4클래스) : {miou_all:.4f}")
    print(f"mIoU (전경 3클래스만)     : {miou_fg:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
