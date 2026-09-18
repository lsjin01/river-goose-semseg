#!/usr/bin/env python3
"""Pinned single-GPU launcher for the expanded-data seven-class baseline."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import yaml


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    os.chdir(repo)
    config_path = repo / 'config/labeling_merged_algae_7class_v2.yaml'
    config = yaml.safe_load(config_path.read_text())
    if not (
        config['data_path'] == 'data/labeling_merged_algae_7class_v2_geo'
        and config['num_classes'] == 7
        and config['segmentation_taxonomy'] == 'merged_algae_7class'
        and config['batch_size'] == 4
        and config['grad_accum_steps'] == 3
        and not config.get('data_parallel', False)
    ):
        raise ValueError('Unexpected seven-class configuration drift')

    data = repo / config['data_path']
    manifest = data / 'manifest.json'
    verification_path = data / 'verification.json'
    verification = json.loads(verification_path.read_text())
    if verification['status'] != 'passed' or verification['manifest_sha256'] != digest(manifest):
        raise ValueError('Dataset verification failed or manifest changed')
    if verification['unsupported_eval_classes'] != ['river']:
        raise ValueError('The frozen river evaluation limitation changed')
    smoke_path = repo / 'outputs/labeling_merged_algae_7class_v2_checks/merged_algae7_seed42.json'
    smoke = json.loads(smoke_path.read_text())
    if not (
        smoke['status'] == 'passed'
        and smoke['config_sha256'] == digest(config_path)
        and smoke['num_classes'] == 7
        and smoke['batch_shape'] == [4, 9, 768, 768]
    ):
        raise ValueError('Full-model batch-4 smoke test is missing or stale')

    output = repo / config['output_dir']
    run_dir = output / config['run_name']
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = [
            config_path,
            manifest,
            verification_path,
            smoke_path,
            repo / 'train.py',
            Path(__file__).resolve(),
            repo / 'tools/prepare_merged_algae_7class.py',
            repo / 'tools/check_fusion_v2.py',
        ]
        files += sorted((repo / 'goose_semseg').rglob('*.py'))
        files += sorted((repo / 'tests').glob('test_*.py'))
        hashes = {str(path.relative_to(repo)): digest(path) for path in files}
        pin = output / 'run_inputs.json'
        if pin.exists():
            if json.loads(pin.read_text()) != hashes:
                raise ValueError('Frozen run inputs changed; use a new run version')
        else:
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError('Unpinned output already exists')
            pin.write_text(json.dumps(hashes, indent=2))
            with tarfile.open(output / 'source_snapshot.tar.gz', 'w:gz') as archive:
                for path in files:
                    archive.add(path, arcname=str(path.relative_to(repo)))
        if (run_dir / 'training_complete.json').exists():
            print('Already completed', flush=True)
            return
        if shutil.disk_usage(repo).free < 15 * 1024**3:
            raise RuntimeError('Less than 15 GiB free')

        def event(status: str, **kwargs) -> None:
            payload = dict(status=status, time=datetime.now(timezone.utc).isoformat(), **kwargs)
            (output / 'state.json').write_text(json.dumps(payload, indent=2))
            with (output / 'status.jsonl').open('a') as stream:
                stream.write(json.dumps(payload) + '\n')
            print(json.dumps(payload), flush=True)

        environment = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES='1',
            PYTHONUNBUFFERED='1',
            PYTHONPATH=str(repo) + ':' + str(repo / 'third_party'),
            OMP_NUM_THREADS='4',
            HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',
            PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        )
        command = [sys.executable, 'train.py', '--config', str(config_path)]
        if (run_dir / 'latest.pt').exists():
            command += ['--resume_from', str(run_dir / 'latest.pt')]
        event(
            'running',
            command=command,
            physical_gpu_ids=[1],
            effective_batch_size=config['batch_size'] * config['grad_accum_steps'],
            initialization='External DINOv3/Mask2Former pretraining; fresh 7-class output layer',
            test_policy='No test evaluation during training',
            known_limitation='river has no validation/test support',
        )
        try:
            with (output / 'train.log').open('a') as log:
                result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f'Trainer exit {result.returncode}')
            complete = json.loads((run_dir / 'training_complete.json').read_text())
            event('completed', best_val_miou=complete['best_val_miou'])
        except Exception as exc:
            event('failed', error=str(exc))
            raise


if __name__ == '__main__':
    main()
