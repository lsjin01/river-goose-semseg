#!/usr/bin/env python3
"""Fail-fast audit for running a YAML experiment on a different machine."""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party')]

from goose_semseg.data.spectral_tiles import resolve_manifest_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--init-from', default=None,
                        help='Portable override for a checkpoint path stored in the YAML.')
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    checks = []

    for module in ('torch', 'torchvision', 'transformers', 'timm', 'numpy', 'PIL', 'yaml'):
        imported = importlib.import_module(module)
        checks.append({'check': f'import:{module}', 'ok': True,
                       'version': getattr(imported, '__version__', None)})

    import torch
    checks.append({'check': 'cuda_available', 'ok': torch.cuda.is_available(),
                   'device_count': torch.cuda.device_count(), 'torch_cuda': torch.version.cuda})
    try:
        import MultiScaleDeformableAttention  # noqa: F401
        checks.append({'check': 'cuda_extension:MultiScaleDeformableAttention', 'ok': True})
    except ImportError as exc:
        checks.append({'check': 'cuda_extension:MultiScaleDeformableAttention', 'ok': False,
                       'error': str(exc)})

    data_root = (ROOT / config['data_path']).resolve()
    manifest_path = data_root / 'manifest.json'
    checks.append({'check': 'manifest', 'ok': manifest_path.is_file(), 'path': str(manifest_path)})
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        try:
            raw_source = resolve_manifest_source(data_root, manifest['source'])
            checks.append({'check': 'raw_source', 'ok': True, 'path': str(raw_source)})
        except FileNotFoundError as exc:
            checks.append({'check': 'raw_source', 'ok': False, 'error': str(exc)})

    checkpoint_value = args.init_from or config.get('init_from')
    if checkpoint_value:
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        checks.append({'check': 'init_checkpoint', 'ok': checkpoint.is_file(),
                       'path': str(checkpoint.resolve())})

    try:
        from transformers import AutoConfig, Mask2FormerConfig
        AutoConfig.from_pretrained(
            config['hf_dinov3_model_name_or_path'], trust_remote_code=True,
            local_files_only=True,
        )
        Mask2FormerConfig.from_pretrained(
            config['mask2former_pretrained_model_name_or_path'], local_files_only=True,
        )
        checks.append({'check': 'hf_model_cache', 'ok': True})
    except Exception as exc:
        checks.append({'check': 'hf_model_cache', 'ok': False, 'error': str(exc)})

    ok = all(check['ok'] for check in checks)
    print(json.dumps({'status': 'passed' if ok else 'failed', 'checks': checks}, indent=2))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
