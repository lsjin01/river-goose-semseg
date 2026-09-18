#!/usr/bin/env python3
"""Two-rank smoke test for full-frame overlap validation."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'third_party')]

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from goose_semseg.data.spectral_tiles import OverlapSpectralTileDataset
from goose_semseg.engine.overlap_validation import evaluate_overlap
from goose_semseg.losses.m2f_criterion import Mask2FormerSetCriterion
from tools.evaluate_overlap_test import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--frames-per-rank', type=int, default=2)
    args = parser.parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', device_id=device)

    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    training = argparse.Namespace(**payload['args'])
    model = build_model(training, payload['model_state_dict'], device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    criterion = Mask2FormerSetCriterion(
        num_classes=training.num_classes,
        no_object_weight=training.m2f_no_object_weight,
        class_weight=training.m2f_class_weight,
        mask_weight=training.m2f_mask_weight,
        dice_weight=training.m2f_dice_weight,
        num_points=training.m2f_train_num_points,
        oversample_ratio=training.m2f_oversample_ratio,
        importance_sample_ratio=training.m2f_importance_sample_ratio,
        ignore_index=training.ignore_index,
        classification_loss_type=training.classification_loss_type,
    ).to(device)
    record_indices = [rank + world_size * index for index in range(args.frames_per_rank)]
    dataset = OverlapSpectralTileDataset(
        ROOT / training.data_path,
        'val',
        training.tile_size,
        512,
        record_indices=record_indices,
    )
    result = evaluate_overlap(
        model=model,
        dataset=dataset,
        criterion=criterion,
        device=device,
        batch_size=8,
        num_workers=2,
        prefetch_factor=1,
        num_classes=training.num_classes,
        ignore_index=training.ignore_index,
        tile_size=training.tile_size,
        epoch=0,
        epochs=1,
    )
    if rank == 0:
        _, miou, _, _, matrix, groups = result
        sensor_sum = groups['sensor/Altum'] + groups['sensor/P1']
        if not torch.equal(matrix, sensor_sum):
            raise RuntimeError('Sensor matrices do not sum to overall overlap validation matrix')
        print(f'DDP OVERLAP SMOKE PASSED frames={world_size * args.frames_per_rank} miou={miou:.6f}')
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
