#!/usr/bin/env python3
"""Padding-free overlapping test evaluation with center-weighted score blending."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party')]

import torch
from torch.utils.data import DataLoader

from goose_semseg.data.spectral_tiles import SpectralTileDataset
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import BackboneLayersSet, build_segmentation_decoder
from goose_semseg.models.head_utils import mask2former_semantic_scores
from goose_semseg.pretrained.mask2former import (
    infer_pretrained_feature_channels,
    load_pretrained_mask2former,
)
from goose_semseg.utils.metrics import compute_mean_iou, per_class_metric_rows, update_confusion_matrix
from goose_semseg.utils.seed import seed_everything


def aligned_positions(length: int, tile_size: int, stride: int) -> list[int]:
    """Cover an axis without padding and force the final window against the far edge."""
    if length < tile_size:
        raise ValueError(f'Image axis {length} is smaller than tile size {tile_size}')
    positions = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions


class OverlapSpectralTileDataset(SpectralTileDataset):
    def __init__(self, root, split: str, tile_size: int, stride: int):
        if split == 'train':
            raise ValueError('Overlap evaluation dataset is not a training sampler')
        if not 0 < stride < tile_size:
            raise ValueError('Overlap stride must lie in (0, tile_size)')
        super().__init__(root, split, tile_size)
        self.stride = int(stride)
        self.tiles = []
        for frame_index, record in enumerate(self.records):
            tops = aligned_positions(record['height'], self.tile_size, self.stride)
            lefts = aligned_positions(record['width'], self.tile_size, self.stride)
            self.tiles.extend((frame_index, left, top) for top in tops for left in lefts)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def clean_rows(matrix: torch.Tensor, names: list[str]) -> list[dict]:
    rows = per_class_metric_rows(matrix, 0)
    for row in rows:
        row.pop('epoch', None)
        row['class_name'] = names[row['class_id']]
        for key, value in row.items():
            if isinstance(value, float) and not math.isfinite(value):
                row[key] = None
    return rows


def metric(matrix: torch.Tensor, names: list[str]) -> dict:
    return {
        'miou_gt_supported': compute_mean_iou(matrix, gt_present_only=True),
        'miou_union_present': compute_mean_iou(matrix),
        'supported_class_ids': torch.where(matrix.sum(1) > 0)[0].tolist(),
        'valid_pixels': int(matrix.sum()),
        'per_class': clean_rows(matrix, names),
        'confusion_matrix': matrix.tolist(),
    }


def build_model(training: argparse.Namespace, state: dict, device: torch.device):
    backbone = load_dinov3_backbone(training)
    pretrained = load_pretrained_mask2former(training.mask2former_pretrained_model_name_or_path)
    feature_channels = infer_pretrained_feature_channels(pretrained)
    model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type='m2f',
        hidden_dim=training.hidden_dim,
        num_classes=training.num_classes,
        autocast_dtype=torch.bfloat16,
        freeze_backbone=training.freeze_backbone,
        feature_channels=feature_channels,
        input_channels=training.input_channels,
        cls_aux_num_classes=(training.cls_aux_num_classes if training.enable_cls_aux else 0),
        fusion_type=training.fusion_type,
        precomputed_indices=True,
    )
    del pretrained
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stride', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    payload = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    training = argparse.Namespace(**payload['args'])
    if not training.source_tiles:
        raise ValueError('Overlap evaluator requires source-grid tiles')
    manifest_path = (ROOT / training.data_path / 'manifest.json').resolve()
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    expected_hash = (checkpoint_path.parent / 'dataset_manifest_sha256.txt').read_text().strip()
    if manifest_hash != expected_hash:
        raise ValueError('Dataset manifest differs from the training checkpoint')

    device = torch.device('cuda')
    seed_everything(training.seed)
    dataset = OverlapSpectralTileDataset(
        ROOT / training.data_path, 'test', training.tile_size, args.stride
    )
    class_names = dataset.manifest['class_names']
    if len(class_names) != training.num_classes:
        raise ValueError('Manifest/model class count mismatch')
    model = build_model(training, payload['model_state_dict'], device)
    selected_epoch = int(payload['epoch']) + 1
    del payload
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=1,
    )

    axis = torch.arange(training.tile_size, device=device, dtype=torch.float32) + 0.5
    sine = torch.sin(torch.pi * axis / training.tile_size)
    blend = (0.1 + 0.9 * torch.outer(sine, sine)).unsqueeze(0)
    matrices = {
        name: torch.zeros(training.num_classes, training.num_classes, dtype=torch.int64, device=device)
        for name in ('overall', 'Altum', 'P1')
    }
    current_frame = None
    score_sum = weight_sum = None
    offset = 0
    completed_frames = 0
    progress = output.with_name(output.stem + '_progress.json')

    def finalize(frame_index: int) -> None:
        nonlocal score_sum, weight_sum, completed_frames
        record = dataset.records[frame_index]
        if torch.any(weight_sum <= 0):
            raise RuntimeError(f'Incomplete overlap coverage for frame {frame_index}')
        prediction = (score_sum / weight_sum).argmax(0)
        target = torch.from_numpy(dataset._mask(frame_index)).to(device=device, dtype=torch.int64)
        update_confusion_matrix(
            matrices['overall'], prediction, target, training.num_classes, training.ignore_index
        )
        update_confusion_matrix(
            matrices[record['sensor']], prediction, target, training.num_classes, training.ignore_index
        )
        completed_frames += 1
        score_sum = weight_sum = None

    print(
        f'OVERLAP TEST START frames={len(dataset.records)} tiles={len(dataset)} '
        f'tile={training.tile_size} stride={args.stride}',
        flush=True,
    )
    with torch.inference_mode():
        for images, _targets in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                scores = mask2former_semantic_scores(
                    model(images), target_size=(training.tile_size, training.tile_size)
                )
            for batch_index in range(len(images)):
                frame_index, left, top = dataset.tiles[offset + batch_index]
                if current_frame != frame_index:
                    if current_frame is not None:
                        finalize(current_frame)
                    current_frame = frame_index
                    record = dataset.records[frame_index]
                    score_sum = torch.zeros(
                        (training.num_classes, record['height'], record['width']),
                        dtype=torch.float32,
                        device=device,
                    )
                    weight_sum = torch.zeros(
                        (1, record['height'], record['width']),
                        dtype=torch.float32,
                        device=device,
                    )
                score_sum[:, top:top + training.tile_size, left:left + training.tile_size].add_(
                    scores[batch_index] * blend
                )
                weight_sum[:, top:top + training.tile_size, left:left + training.tile_size].add_(blend)
            offset += len(images)
            if offset % 500 < len(images) or offset == len(dataset):
                elapsed = time.monotonic() - started
                status = {
                    'status': 'running',
                    'tiles_done': offset,
                    'total_tiles': len(dataset),
                    'frames_done': completed_frames,
                    'total_frames': len(dataset.records),
                    'elapsed_seconds': elapsed,
                    'estimated_remaining_seconds': elapsed / offset * (len(dataset) - offset),
                }
                atomic_json(progress, status)
                print(json.dumps(status), flush=True)
    if current_frame is not None:
        finalize(current_frame)
    if offset != len(dataset) or completed_frames != len(dataset.records):
        raise RuntimeError('Incomplete test traversal')
    if not torch.equal(matrices['overall'], matrices['Altum'] + matrices['P1']):
        raise RuntimeError('Sensor confusion matrices do not sum to overall')

    result = {
        'status': 'completed',
        'started_utc': started_utc,
        'completed_utc': datetime.now(timezone.utc).isoformat(),
        'checkpoint': str(checkpoint_path.relative_to(ROOT)),
        'checkpoint_epoch': selected_epoch,
        'split': 'test',
        'tta': False,
        'frames': len(dataset.records),
        'tiles': len(dataset),
        'class_names': class_names,
        'manifest_sha256': manifest_hash,
        'evaluation': {
            'tile_size': training.tile_size,
            'stride': args.stride,
            'overlap_pixels': training.tile_size - args.stride,
            'edge_aligned': True,
            'padding': False,
            'blend': 'semantic-score weighted mean; weight=0.1+0.9*sin(pi*x/T)*sin(pi*y/T)',
            'metric_definition': 'mIoU over GT-supported classes; river absent from test GT',
        },
        'metrics': {
            name: metric(matrix.cpu(), class_names) for name, matrix in matrices.items()
        },
        'elapsed_seconds': time.monotonic() - started,
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'peak_cuda_allocated_bytes': [
            torch.cuda.max_memory_allocated(index) for index in range(torch.cuda.device_count())
        ],
    }
    atomic_json(output, result)
    atomic_json(progress, {'status': 'completed', 'output': str(output), 'tiles_done': offset})
    print('OVERLAP TEST COMPLETE', result['metrics']['overall']['miou_gt_supported'], flush=True)


if __name__ == '__main__':
    main()
