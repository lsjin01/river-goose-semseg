#!/usr/bin/env python3
"""Wait for the current student KD queue, then fine-tune the multi-river teacher."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/conda/envs/goose/bin/python")
UPSTREAM_STATE = ROOT / "outputs/labeling_merged_algae_7class_kd/queue_state.json"
CONFIG = ROOT / "config/labeling_merged_algae_7class_multiriver_ft_gpu1.yaml"
SOURCE = ROOT / "outputs_old/baseline_100ep/baseline_cat2_100ep/best.pt"
MANIFEST = ROOT / "data/labeling_merged_algae_7class_v2_geo/manifest.json"
OUTPUT_ROOT = ROOT / "outputs/labeling_merged_algae_7class_multiriver_ft_gpu1"
RUN_DIR = OUTPUT_ROOT / "merged_algae7_multiriver_ft_seed42"
STATE = OUTPUT_ROOT / "queue_state.json"


def write_state(**values) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"updated_utc": datetime.now(timezone.utc).isoformat(), **values}
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(STATE)
    print(json.dumps(payload), flush=True)


def wait_for_upstream() -> None:
    while True:
        if UPSTREAM_STATE.is_file():
            upstream = json.loads(UPSTREAM_STATE.read_text())
            if upstream.get("status") == "completed":
                return
            if upstream.get("status") == "failed":
                raise RuntimeError(f"Upstream KD queue failed: {upstream}")
        write_state(status="waiting_for_b0_kd", upstream=str(UPSTREAM_STATE.relative_to(ROOT)))
        time.sleep(30)


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("This pinned queue must run with CUDA_VISIBLE_DEVICES=1")
    for required in (CONFIG, SOURCE, MANIFEST):
        if not required.is_file():
            raise FileNotFoundError(required)
    if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
        raise FileExistsError(f"Refusing to overwrite fine-tune output: {RUN_DIR}")
    wait_for_upstream()
    write_state(
        status="training",
        source_checkpoint=str(SOURCE.relative_to(ROOT)),
        config=str(CONFIG.relative_to(ROOT)),
        manifest_sha256=hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        initialization="multi-river DINOv3+Mask2Former; new spectral adapter and 7-class head",
    )
    subprocess.run(
        [str(PYTHON), "train.py", "--config", str(CONFIG.relative_to(ROOT))],
        cwd=ROOT,
        env=os.environ.copy(),
        check=True,
    )
    write_state(status="completed", output=str(RUN_DIR.relative_to(ROOT)))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_state(status="failed", error=repr(exc))
        raise
