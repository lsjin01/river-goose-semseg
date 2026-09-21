#!/usr/bin/env python3
"""Wait for teacher test evaluation, then train SegFormer-B1 and B0 on GPU 1."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
TEACHER = Path(os.environ.get(
    "RIVER_SEMSEG_TEACHER_CHECKPOINT",
    ROOT / "outputs/labeling_merged_algae_7class_v3_gpu1/merged_algae7_gpu1_overlap_seed42/best_epoch_075_miou_0.6207.pt",
)).expanduser()
if not TEACHER.is_absolute():
    TEACHER = ROOT / TEACHER
TEST_RESULT = TEACHER.parent / "test_overlap_stride512.json"
QUEUE_ROOT = ROOT / "outputs/labeling_merged_algae_7class_kd"
STATE = QUEUE_ROOT / "queue_state.json"


def write_state(**values) -> None:
    QUEUE_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"updated_utc": datetime.now(timezone.utc).isoformat(), **values}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATE)
    print(json.dumps(payload), flush=True)


def wait_for_test() -> None:
    while True:
        if TEST_RESULT.is_file():
            payload = json.loads(TEST_RESULT.read_text())
            if payload.get("status") != "completed" or payload.get("split") != "test":
                raise RuntimeError(f"Invalid teacher test result: {TEST_RESULT}")
            return
        write_state(status="waiting_for_teacher_test", test_result=str(TEST_RESULT.relative_to(ROOT)))
        time.sleep(30)


def train(model: str) -> None:
    short = model.rsplit("_", 1)[-1]
    output = QUEUE_ROOT / short
    if output.exists() and any(output.iterdir()):
        progress = output / "progress.json"
        if progress.is_file() and json.loads(progress.read_text()).get("status") == "completed":
            write_state(status="skipped_completed", student_model=model, output=str(output.relative_to(ROOT)))
            return
        raise FileExistsError(f"Refusing to overwrite incomplete KD output: {output}")
    write_state(status="training", student_model=model, output=str(output.relative_to(ROOT)))
    command = [
        str(PYTHON), "tools/train_spectral_student_kd.py",
        "--teacher-checkpoint", str(TEACHER),
        "--student-model", model,
        "--data-path", "data/labeling_merged_algae_7class_v2_geo",
        "--output-dir", str(output),
        "--epochs", "50",
        "--batch-size", "2",
        "--grad-accum-steps", "6",
        "--num-workers", "4",
        "--tile-size", "768",
        "--val-stride", "512",
        "--val-batch-size", "8",
        "--val-interval", "5",
        "--early-stopping-patience", "4",
        "--lambda-kd", "1.0",
        "--kd-temperature", "4.0",
        "--class-sampling-prob", "0.3",
        "--student-local-files-only",
    ]
    subprocess.run(command, cwd=ROOT, env=os.environ.copy(), check=True)


def main() -> None:
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the one physical GPU used by this queue")
    if not TEACHER.is_file():
        raise FileNotFoundError(TEACHER)
    wait_for_test()
    for model in ("segformer_b1", "segformer_b0"):
        train(model)
    write_state(status="completed", models=["segformer_b1", "segformer_b0"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_state(status="failed", error=repr(exc))
        raise
