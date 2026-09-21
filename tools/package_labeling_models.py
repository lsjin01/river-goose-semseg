#!/usr/bin/env python3
"""Collect completed Labeling_Data checkpoints under descriptive portable names."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "artifacts/labeling_data_v2_7class_models"
MODELS = (
    {
        "role": "teacher",
        "architecture": "dinov3_vitl16_mask2former",
        "source": ROOT / "outputs/labeling_merged_algae_7class_v3_gpu1/merged_algae7_gpu1_overlap_seed42/best_epoch_075_miou_0.6207.pt",
        "filename": "labeling_v2_7class_teacher_dinov3_vitl16_mask2former_best_val_miou_0.6207.pt",
        "test_result": ROOT / "outputs/labeling_merged_algae_7class_v3_gpu1/merged_algae7_gpu1_overlap_seed42/test_overlap_stride512.json",
    },
    {
        "role": "student",
        "architecture": "segformer_b1_kd",
        "source": ROOT / "outputs/labeling_merged_algae_7class_kd/b1/best.pt",
        "filename": "labeling_v2_7class_student_segformer_b1_kd_best_val_miou_0.4814.pt",
    },
    {
        "role": "student",
        "architecture": "segformer_b0_kd",
        "source": ROOT / "outputs/labeling_merged_algae_7class_kd/b0/best.pt",
        "filename": "labeling_v2_7class_student_segformer_b0_kd_best_val_miou_0.4576.pt",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def publish_hardlink(source: Path, destination: Path) -> None:
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(f"Refusing to replace existing artifact: {destination}")
    os.link(source, destination)


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for spec in MODELS:
        source = spec["source"]
        if not source.is_file():
            raise FileNotFoundError(source)
        checkpoint = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
        target = DESTINATION / spec["filename"]
        publish_hardlink(source, target)
        if spec["role"] == "teacher":
            best_val = float(checkpoint["best_val_miou"])
            epoch = int(checkpoint["epoch"]) + 1
        else:
            best_val = float(checkpoint["val_metrics"]["overall"]["miou_gt_supported"])
            epoch = int(checkpoint["epoch"])
        record = {
            "filename": target.name,
            "role": spec["role"],
            "architecture": spec["architecture"],
            "epoch": epoch,
            "best_validation_miou_gt_supported": best_val,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "source_path": str(source.relative_to(ROOT)),
            "storage": "hardlink_to_source_checkpoint",
        }
        test_result = spec.get("test_result")
        if test_result and test_result.is_file():
            result = json.loads(test_result.read_text())
            record["test_miou_gt_supported"] = result["metrics"]["overall"][
                "miou_gt_supported"
            ]
            record["test_tta"] = result["tta"]
        artifacts.append(record)
        del checkpoint

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Labeling_Data_v2",
        "taxonomy": "merged_algae_7class",
        "input_channels": 9,
        "class_names": [
            "river", "land", "bridge", "other", "nps", "turbid",
            "algae_including_nps_algae",
        ],
        "models": artifacts,
        "note": (
            "These are the completed fresh-teacher/KD experiment artifacts. "
            "The later multi-river-initialized teacher fine-tune is still a separate run."
        ),
    }
    manifest_path = DESTINATION / "manifest.json"
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2))
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
