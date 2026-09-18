from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent
for _p in (_REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


import argparse
import json
import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader

from goose_semseg.data.dataset import GooseSegmentationDataset
from goose_semseg.engine.trainer import run_epoch
from goose_semseg.losses.m2f_criterion import Mask2FormerSetCriterion
from goose_semseg.models.backbone.loader import load_dinov3_backbone
from goose_semseg.models.builder import BackboneLayersSet, build_segmentation_decoder
from goose_semseg.optim.optimizer import build_optimizer
from goose_semseg.optim.schedulers import SCHEDULERS_DICT, build_scheduler
from goose_semseg.pretrained.mask2former import (
    infer_pretrained_feature_channels,
    initialize_head_from_pretrained,
    load_pretrained_mask2former,
)
from goose_semseg.utils.checkpoint import (
    load_model_state_allowing_token_specialization,
    save_checkpoint,
)
from goose_semseg.utils.logging import (
    append_epoch_log,
    append_per_class_metrics,
    append_presence_summary,
)
from goose_semseg.utils.samples import (
    _compute_cls_aux_pos_weight_from_samples,
    _load_yaml_config,
    _resolve_cls_aux_output_dim,
)
from goose_semseg.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. CLI arguments override YAML values.",
    )
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = argparse.ArgumentParser("GOOSE DINOv3 Adapter + Mask2Former trainer")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config file. CLI arguments override YAML values.",
    )
    parser.add_argument("--data_path", default="data/goose")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=None,
        help="Validation batch size; defaults to --batch_size.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=1)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--encoder_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--scheduler_type",
        choices=["none"] + list(SCHEDULERS_DICT.keys()),
        default="none",
        help=(
            "Iteration-based LR scheduler. 'PolynomialLR' follows the DeepLab/MMSeg recipe; "
            "'MultiStepLR' the Mask2Former recipe; 'WarmupOneCycleLR' the DINOv3 team's variant. "
            "Set to 'none' to disable."
        ),
    )
    parser.add_argument(
        "--scheduler_kwargs",
        type=str,
        default="{}",
        help=(
            "JSON string of scheduler-specific constructor kwargs, "
            "e.g. '{\"power\": 0.9}' for PolynomialLR or "
            "'{\"milestones\": [54000, 57000], \"gamma\": 0.1}' for MultiStepLR."
        ),
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=0,
        help=(
            "If > 0, prepend a LinearLR warmup of this many iterations before the main "
            "scheduler kicks in (1500 is a common DeepLab default)."
        ),
    )
    parser.add_argument("--grad_clip_norm", type=float, default=35.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--resize_width", type=int, default=1024)
    parser.add_argument("--resize_height", type=int, default=1024)
    parser.add_argument("--num_classes", type=int, default=64)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--data_parallel",
        action="store_true",
        help="Use all CUDA devices visible to this process via torch.nn.DataParallel.",
    )
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--flip_prob", type=float, default=0.0)
    parser.add_argument("--input_channels", type=int, default=3)
    parser.add_argument("--source_tiles", action="store_true")
    parser.add_argument(
        "--segmentation_taxonomy",
        choices=['source', 'binary_algae', 'merged_algae_7class'],
        default='source',
    )
    parser.add_argument("--tile_size", type=int, default=768)
    parser.add_argument("--class_sampling_prob", type=float, default=0.7)
    parser.add_argument("--class_sampling_mode", choices=['global', 'within_image'], default='global')
    parser.add_argument("--val_group_metrics", action='store_true')
    parser.add_argument("--val_interval", type=int, default=1)
    parser.add_argument(
        "--fusion_type",
        choices=["input", "indices", "late", "gated", "cross_attention"],
        default="input",
    )
    parser.add_argument("--enable_random_crop", action="store_true")
    parser.add_argument("--crop_width", type=int, default=1024)
    parser.add_argument("--crop_height", type=int, default=1024)
    parser.add_argument("--enable_rare_class_crop", action="store_true")
    parser.add_argument("--rare_class_crop_prob", type=float, default=0.3)
    parser.add_argument(
        "--rare_class_ids",
        type=int,
        nargs="*",
        default=[],
        help="Fine-class ids used by rare-class-aware cropping.",
    )
    parser.add_argument("--rare_class_min_pixels", type=int, default=0)
    parser.add_argument("--rare_class_min_ratio", type=float, default=0.005)
    parser.add_argument("--rare_class_crop_attempts", type=int, default=10)
    parser.add_argument("--backbone_loader", choices=["auto", "native", "hf"], default="auto")
    parser.add_argument(
        "--hf_dinov3_model_name_or_path",
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    parser.add_argument("--allow_hf_download", dest="hf_local_files_only", action="store_false")
    parser.set_defaults(hf_local_files_only=True)
    parser.add_argument("--dinov3_weights", default=None)
    parser.add_argument(
        "--mask2former_pretrained_model_name_or_path",
        default="facebook/mask2former-swin-large-ade-semantic",
    )
    parser.add_argument("--disable_mask2former_pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--m2f_no_object_weight", type=float, default=0.1)
    parser.add_argument("--m2f_class_weight", type=float, default=2.0)
    parser.add_argument("--m2f_mask_weight", type=float, default=5.0)
    parser.add_argument("--m2f_dice_weight", type=float, default=5.0)
    parser.add_argument("--m2f_train_num_points", type=int, default=12_544)
    parser.add_argument("--m2f_oversample_ratio", type=float, default=3.0)
    parser.add_argument("--m2f_importance_sample_ratio", type=float, default=0.75)
    parser.add_argument("--classification_loss_type", default="ce", choices=["ce", "focal", "seesaw"])
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--init_from", default=None)
    parser.add_argument("--resume_from", default=None)
    parser.add_argument("--enable_cls_aux", action="store_true")
    parser.add_argument("--cls_aux_taxonomy", choices=['legacy', 'labeling_data'], default='legacy',
                        help='Coarse hierarchy: legacy AIHub (3) or new Labeling_Data (7).')
    parser.add_argument(
        "--cls_aux_target_type",
        choices=["coarse", "fine"],
        default="coarse",
        help="CLS auxiliary multi-hot target: taxonomy-specific coarse groups or fine classes.",
    )
    parser.add_argument("--max_best_checkpoints", type=int, default=0)
    parser.add_argument(
        "--cls_aux_num_classes",
        type=int,
        default=None,
        help="Deprecated compatibility option. cls_aux output dim is resolved automatically from cls_aux_target_type.",
    )
    parser.add_argument(
        "--cls_aux_loss_type",
        choices=["bce", "weighted_bce"],
        default="bce",
        help="Loss for CLS auxiliary multi-hot supervision.",
    )
    parser.add_argument("--cls_aux_weight", type=float, default=0.1)
    parser.add_argument(
        "--cls_aux_pos_weight_max",
        type=float,
        default=20.0,
        help="Upper clip for automatic pos_weight when cls_aux_loss_type=weighted_bce.",
    )
    if config_args.config:
        yaml_config = _load_yaml_config(config_args.config)
        valid_keys = {action.dest for action in parser._actions}
        unknown_keys = sorted(set(yaml_config) - valid_keys)
        if unknown_keys:
            raise ValueError(
                "Unknown config keys in {}: {}".format(
                    config_args.config,
                    ", ".join(unknown_keys),
                )
            )
        parser.set_defaults(**yaml_config)
    return parser.parse_args(remaining_argv)


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = "goose_dinov3_vitl16_m2f_ce_1024"
    if args.init_from and args.resume_from:
        raise ValueError("Use either --init_from or --resume_from, not both.")
    if args.segmentation_taxonomy == 'binary_algae':
        if not args.source_tiles or args.num_classes != 2 or args.ignore_index != 255:
            raise ValueError('binary_algae requires source_tiles, num_classes=2, ignore_index=255')
        if args.enable_cls_aux and args.cls_aux_target_type != 'fine':
            raise ValueError('binary_algae CLS auxiliary must use the two remapped fine targets')
    if args.segmentation_taxonomy == 'merged_algae_7class':
        if not args.source_tiles or args.num_classes != 7 or args.ignore_index != 255:
            raise ValueError('merged_algae_7class requires source_tiles, num_classes=7, ignore_index=255')
        if args.enable_cls_aux and args.cls_aux_target_type != 'fine':
            raise ValueError('merged_algae_7class CLS auxiliary must use fine targets')
    args.cls_aux_target_type = str(args.cls_aux_target_type).lower()
    args.cls_aux_num_classes = _resolve_cls_aux_output_dim(
        target_type=args.cls_aux_target_type,
        num_classes=args.num_classes,
        taxonomy=args.cls_aux_taxonomy,
    )
    cls_aux_fine_to_coarse = None
    if args.enable_cls_aux and args.cls_aux_target_type == 'coarse' and args.cls_aux_taxonomy == 'labeling_data':
        from goose_semseg.data.labeling_taxonomy import fine_to_coarse_mapping
        cls_aux_fine_to_coarse = fine_to_coarse_mapping()
    if args.source_tiles and args.enable_cls_aux:
        if args.cls_aux_target_type == 'coarse' and args.cls_aux_taxonomy != 'labeling_data':
            raise ValueError('source_tiles coarse auxiliary requires --cls_aux_taxonomy labeling_data')
        if args.cls_aux_loss_type == 'weighted_bce':
            raise ValueError('source_tiles auxiliary currently supports bce only; '
                             'image-presence weights do not represent sampled crops')
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This trainer expects a CUDA GPU for DINOv3 ViT-L + Mask2Former.")

    run_dir = Path(args.output_dir).expanduser().resolve() / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_args.json").open("w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)

    resize_size = (args.resize_width, args.resize_height)
    if args.source_tiles:
        from goose_semseg.data.spectral_tiles import SpectralTileDataset
        dataset_class = SpectralTileDataset
        if args.segmentation_taxonomy == 'binary_algae':
            from goose_semseg.data.binary_algae import BinaryAlgaeTileDataset, binary_taxonomy_metadata
            dataset_class = BinaryAlgaeTileDataset
            (run_dir/'segmentation_taxonomy.json').write_text(json.dumps(binary_taxonomy_metadata(), indent=2))
        if args.input_channels != 9:
            raise ValueError('source_tiles requires input_channels=9')
        train_dataset = dataset_class(args.data_path, 'train', args.tile_size,
                                           args.flip_prob, args.class_sampling_prob,
                                           args.class_sampling_mode)
        val_dataset = dataset_class(args.data_path, 'val', args.tile_size)
        if args.segmentation_taxonomy == 'binary_algae' and train_dataset.native_binary:
            (run_dir/'segmentation_taxonomy.json').write_text(json.dumps(dict(
                name='binary_algae',class_names=train_dataset.manifest['class_names'],
                annotation_label_key='label_id',ignore_index=255,source_annotations_unchanged=True,
                **train_dataset.manifest['taxonomy']),indent=2))
        if args.segmentation_taxonomy == 'merged_algae_7class':
            from goose_semseg.data.merged_algae import CLASS_NAMES, taxonomy_metadata
            if train_dataset.manifest.get('annotation_label_key') != 'label_id':
                raise ValueError('merged_algae_7class requires a native label_id manifest')
            if train_dataset.manifest['class_names'] != list(CLASS_NAMES):
                raise ValueError('merged_algae_7class manifest class order does not match the model')
            (run_dir/'segmentation_taxonomy.json').write_text(
                json.dumps(taxonomy_metadata(), indent=2)
            )
        if cls_aux_fine_to_coarse is not None:
            from goose_semseg.data.labeling_taxonomy import COARSE_NAMES, fine_to_coarse_mapping
            cls_aux_fine_to_coarse = fine_to_coarse_mapping(train_dataset.manifest['class_names'])
            (run_dir/'cls_aux_taxonomy.json').write_text(json.dumps(dict(
                fine_names=train_dataset.manifest['class_names'], coarse_names=COARSE_NAMES,
                fine_to_coarse=cls_aux_fine_to_coarse), indent=2))
        import hashlib
        manifest_bytes = (Path(args.data_path)/'manifest.json').read_bytes()
        (run_dir/'dataset_manifest_sha256.txt').write_text(hashlib.sha256(manifest_bytes).hexdigest())
    else:
        train_dataset = GooseSegmentationDataset(
            args.data_path,
            "train",
            resize_size=resize_size,
            flip_prob=args.flip_prob,
            enable_random_crop=args.enable_random_crop,
            crop_size=(args.crop_width, args.crop_height),
            enable_rare_class_crop=args.enable_rare_class_crop,
            rare_class_crop_prob=args.rare_class_crop_prob,
            rare_class_ids=args.rare_class_ids,
            rare_class_min_pixels=args.rare_class_min_pixels,
            rare_class_min_ratio=args.rare_class_min_ratio,
            input_channels=args.input_channels,
            rare_class_crop_attempts=args.rare_class_crop_attempts,
        )
        val_dataset = GooseSegmentationDataset(
            args.data_path,
            "val",
            resize_size=resize_size,
            flip_prob=0.0,
            input_channels=args.input_channels,
        )
    print(f"Loaded {len(train_dataset)} train samples and {len(val_dataset)} val samples.")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": True,
        "persistent_workers": False,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=(args.val_batch_size or args.batch_size),
        shuffle=False,
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed + 1),
        **loader_kwargs,
    )

    backbone = load_dinov3_backbone(args)

    pretrained_mask2former = None
    feature_channels = None
    if not args.disable_mask2former_pretrained:
        pretrained_mask2former = load_pretrained_mask2former(
            args.mask2former_pretrained_model_name_or_path
        )
        feature_channels = infer_pretrained_feature_channels(pretrained_mask2former)
        pretrained_hidden_dim = int(pretrained_mask2former.config.hidden_dim)
        if args.hidden_dim != pretrained_hidden_dim:
            print(
                f"Overriding hidden_dim={args.hidden_dim} with pretrained "
                f"Mask2Former hidden_dim={pretrained_hidden_dim}."
            )
            args.hidden_dim = pretrained_hidden_dim

    model = build_segmentation_decoder(
        backbone,
        backbone_out_layers=BackboneLayersSet.FOUR_EVEN_INTERVALS,
        decoder_type="m2f",
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        autocast_dtype=torch.bfloat16,
        freeze_backbone=args.freeze_backbone,
        feature_channels=feature_channels,
        input_channels=args.input_channels,
        cls_aux_num_classes=(args.cls_aux_num_classes if args.enable_cls_aux else 0),
        fusion_type=args.fusion_type,
        precomputed_indices=args.source_tiles,
    ).to(device)
    if args.enable_cls_aux:
        print(
            "CLS auxiliary enabled: "
            f"target_type={args.cls_aux_target_type} output_dim={args.cls_aux_num_classes} "
            f"loss_type={args.cls_aux_loss_type}"
        )
    if pretrained_mask2former is not None:
        initialize_head_from_pretrained(model, pretrained_mask2former)
    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device)
        missing_keys, unexpected_keys, _ = load_model_state_allowing_token_specialization(
            model,
            checkpoint["model_state_dict"],
        )
        if missing_keys:
            print(
                "Warning: init_from left {} model tensors initialized from the "
                "current model. First keys: {}".format(
                    len(missing_keys),
                    ", ".join(missing_keys[:5]),
                )
            )
        if unexpected_keys:
            print(
                "Warning: init_from had {} unused tensors. First keys: {}".format(
                    len(unexpected_keys),
                    ", ".join(unexpected_keys[:5]),
                )
            )
        print(f"Initialized model weights from {args.init_from}.")

    criterion = Mask2FormerSetCriterion(
        num_classes=args.num_classes,
        no_object_weight=args.m2f_no_object_weight,
        class_weight=args.m2f_class_weight,
        mask_weight=args.m2f_mask_weight,
        dice_weight=args.m2f_dice_weight,
        num_points=args.m2f_train_num_points,
        oversample_ratio=args.m2f_oversample_ratio,
        importance_sample_ratio=args.m2f_importance_sample_ratio,
        ignore_index=args.ignore_index,
        classification_loss_type=args.classification_loss_type,
    ).to(device)

    optimizer = build_optimizer(
        model,
        lr=args.lr,
        encoder_lr=args.encoder_lr,
        weight_decay=args.weight_decay,
    )

    iters_per_epoch = math.ceil(len(train_loader) / max(1, int(args.grad_accum_steps)))
    total_iter = iters_per_epoch * int(args.epochs)
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    if str(args.scheduler_type).lower() != "none":
        constructor_kwargs = json.loads(args.scheduler_kwargs) if args.scheduler_kwargs else {}
        warmup_iters = max(0, int(args.warmup_iters))
        main_total_iter = max(1, total_iter - warmup_iters)
        main_scheduler = build_scheduler(
            scheduler_type=args.scheduler_type,
            optimizer=optimizer,
            lr=float(args.lr),
            total_iter=main_total_iter,
            constructor_kwargs=constructor_kwargs,
        )
        if warmup_iters > 0:
            warmup = LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_iters,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, main_scheduler],
                milestones=[warmup_iters],
            )
        else:
            scheduler = main_scheduler
        print(
            f"LR scheduler enabled: {args.scheduler_type} "
            f"total_iter={total_iter} warmup_iters={warmup_iters} "
            f"kwargs={constructor_kwargs}"
        )
    scaler = torch.amp.GradScaler(device.type, enabled=(not args.no_amp))
    cls_aux_pos_weight: Optional[torch.Tensor] = None
    if args.enable_cls_aux and args.cls_aux_loss_type == "weighted_bce":
        cls_aux_pos_weight = _compute_cls_aux_pos_weight_from_samples(
            samples=train_dataset.samples,
            target_type=args.cls_aux_target_type,
            num_classes=args.num_classes,
            cls_aux_num_classes=args.cls_aux_num_classes,
            ignore_index=args.ignore_index,
            max_pos_weight=args.cls_aux_pos_weight_max,
            fine_to_coarse=cls_aux_fine_to_coarse,
        )
        preview_count = min(8, cls_aux_pos_weight.numel())
        preview = ", ".join(f"{float(value):.2f}" for value in cls_aux_pos_weight[:preview_count])
        print(
            "Computed cls_aux weighted BCE pos_weight "
            f"(first {preview_count}/{cls_aux_pos_weight.numel()}): {preview}"
        )
    start_epoch = 0
    best_val_miou = float("-inf")
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        missing_keys, unexpected_keys, _ = load_model_state_allowing_token_specialization(
            model,
            checkpoint["model_state_dict"],
        )
        if missing_keys:
            print(
                "Warning: checkpoint load left {} model tensors initialized from "
                "the current model. First keys: {}".format(
                    len(missing_keys),
                    ", ".join(missing_keys[:5]),
                )
            )
        if unexpected_keys:
            print(
                "Warning: checkpoint had {} unused tensors. First keys: {}".format(
                    len(unexpected_keys),
                    ", ".join(unexpected_keys[:5]),
                )
            )
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (RuntimeError, ValueError) as exc:
            print(
                "Warning: skipped optimizer state loading because the parameter "
                f"groups do not match the current model. Original error: {exc}"
            )
        try:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        except (RuntimeError, ValueError) as exc:
            print(f"Warning: skipped scaler state loading. Original error: {exc}")
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("Restored scheduler state from checkpoint.")
            except (RuntimeError, ValueError) as exc:
                print(f"Warning: skipped scheduler state loading. Original error: {exc}")
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_miou = float(checkpoint.get("best_val_miou", best_val_miou))
        print(f"Resumed from {args.resume_from} at epoch {start_epoch}.")

    if args.data_parallel:
        visible_device_count = torch.cuda.device_count()
        if visible_device_count < 2:
            raise RuntimeError(
                "--data_parallel requires at least two visible CUDA devices; "
                f"found {visible_device_count}."
            )
        model = torch.nn.DataParallel(model)
        print(f"DataParallel enabled across {visible_device_count} CUDA devices.")

    epochs_without_improvement = 0
    if args.resume_from:
        import csv
        metrics_file = run_dir/'epoch_metrics.csv'
        if metrics_file.exists():
            history = list(csv.DictReader(metrics_file.open()))
            previous_best, last_improved = float('-inf'), 0
            for row in history:
                if int(row['epoch']) > start_epoch:
                    continue
                if float(row['val_miou']) > previous_best + args.early_stopping_min_delta:
                    previous_best, last_improved = float(row['val_miou']), int(row['epoch'])
            epochs_without_improvement = sum(last_improved < int(row['epoch']) <= start_epoch for row in history)
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_miou, train_breakdown, train_cls_metrics, train_confusion_matrix = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=not args.no_amp,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=args.grad_accum_steps,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            epoch=epoch,
            epochs=args.epochs,
            enable_cls_aux=args.enable_cls_aux,
            cls_aux_target_type=args.cls_aux_target_type,
            cls_aux_loss_type=args.cls_aux_loss_type,
            cls_aux_weight=args.cls_aux_weight,
            cls_aux_num_classes=args.cls_aux_num_classes,
            cls_aux_pos_weight=cls_aux_pos_weight,
            cls_aux_fine_to_coarse=cls_aux_fine_to_coarse,
            iter_scheduler=scheduler,
            gt_present_only=args.source_tiles,
        )
        if args.val_group_metrics:
            append_per_class_metrics(run_dir/'train_per_class_metrics.csv', epoch, train_confusion_matrix)
        if (epoch + 1) % max(1, args.val_interval) != 0 and epoch + 1 != args.epochs:
            print(f'epoch={epoch+1} train_loss={train_loss:.4f} train_miou={train_miou:.4f} validation=scheduled')
            progress = dict(epoch=epoch+1, train_loss=train_loss, train_miou=train_miou,
                            best_val_miou=best_val_miou)
            if args.enable_cls_aux:
                progress.update(train_cls_aux_loss=train_breakdown['loss_cls_aux'],
                                train_cls_aux_accuracy=train_cls_metrics['cls_aux_accuracy'])
                print(f"  train_cls_aux_loss={train_breakdown['loss_cls_aux']:.6f} "
                      f"train_cls_aux_acc={train_cls_metrics['cls_aux_accuracy']:.4f}")
            (run_dir/'progress.json').write_text(json.dumps(progress))
            save_checkpoint(run_dir/'latest.pt',model=model,optimizer=optimizer,scaler=scaler,
                scheduler=scheduler,epoch=epoch,best_val_miou=best_val_miou,args=args)
            continue
        val_groups = {} if args.val_group_metrics and args.source_tiles else None
        val_loss, val_miou, val_breakdown, val_cls_metrics, val_confusion_matrix = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=scaler,
            device=device,
            amp=not args.no_amp,
            grad_clip_norm=args.grad_clip_norm,
            grad_accum_steps=1,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            epoch=epoch,
            epochs=args.epochs,
            enable_cls_aux=args.enable_cls_aux,
            cls_aux_target_type=args.cls_aux_target_type,
            cls_aux_loss_type=args.cls_aux_loss_type,
            cls_aux_weight=args.cls_aux_weight,
            cls_aux_num_classes=args.cls_aux_num_classes,
            cls_aux_pos_weight=cls_aux_pos_weight,
            cls_aux_fine_to_coarse=cls_aux_fine_to_coarse,
            gt_present_only=args.source_tiles,
            group_confusion_matrices=val_groups,
        )

        if val_groups is not None:
            from goose_semseg.utils.metrics import compute_mean_iou, per_class_metric_rows
            val_groups['overall'] = val_confusion_matrix
            grouped = {}
            for name, matrix in val_groups.items():
                rows = per_class_metric_rows(matrix, epoch)
                for row in rows:
                    row['class_name'] = val_dataset.manifest['class_names'][row['class_id']]
                    for key, value in row.items():
                        if isinstance(value, float) and not math.isfinite(value):
                            row[key] = None
                grouped[name] = dict(miou=compute_mean_iou(matrix, gt_present_only=True),
                    supported_class_ids=torch.where(matrix.sum(1)>0)[0].tolist(),
                    confusion_matrix=matrix.tolist(), per_class=rows)
            (run_dir/f'val_groups_epoch_{epoch+1:03d}.json').write_text(json.dumps(dict(
                epoch=epoch+1, tta=False, metrics=grouped,
                metric_definition='Each group averages its GT-supported classes; compare identical groups across runs.'
            ), indent=2, allow_nan=False))

        log_message = (
            f"epoch={epoch + 1} train_loss={train_loss:.4f} train_miou={train_miou:.4f} "
            f"val_loss={val_loss:.4f} val_miou={val_miou:.4f} best_val_miou={best_val_miou:.4f}"
        )
        if args.enable_cls_aux:
            log_message += (
                f" train_cls_aux_acc={train_cls_metrics['cls_aux_accuracy']:.4f}"
                f" val_cls_aux_acc={val_cls_metrics['cls_aux_accuracy']:.4f}"
            )
        print(log_message)
        if scheduler is not None:
            current_lrs = [group["lr"] for group in optimizer.param_groups]
            print(f"  current_lrs={current_lrs}")

        improved = val_miou > best_val_miou + args.early_stopping_min_delta
        if improved:
            best_val_miou = val_miou
            epochs_without_improvement = 0
            best_checkpoint_path = run_dir / f"best_epoch_{epoch + 1:03d}_miou_{val_miou:.4f}.pt"
            save_checkpoint(
                best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                best_val_miou=best_val_miou,
                args=args,
            )
            if args.max_best_checkpoints > 0:
                previous_best = sorted(
                    path for path in run_dir.glob("best_epoch_*.pt") if path != best_checkpoint_path
                )
                for obsolete in previous_best[:max(0, len(previous_best) - args.max_best_checkpoints + 1)]:
                    obsolete.unlink()
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            run_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            best_val_miou=best_val_miou,
            args=args,
        )
        append_epoch_log(
            run_dir / "epoch_metrics.csv",
            epoch,
            train_loss,
            train_miou,
            val_loss,
            val_miou,
            best_val_miou,
            train_breakdown,
            val_breakdown,
            train_cls_metrics,
            val_cls_metrics,
        )
        append_per_class_metrics(
            run_dir / "val_per_class_metrics.csv",
            epoch,
            val_confusion_matrix,
        )
        append_presence_summary(
            run_dir / "val_presence_summary.csv",
            epoch,
            val_confusion_matrix,
        )
        if args.source_tiles:
            (run_dir/'progress.json').write_text(json.dumps(dict(epoch=epoch+1,
                train_loss=train_loss, train_miou=train_miou, val_loss=val_loss,
                val_miou=val_miou, best_val_miou=best_val_miou)))

        if epochs_without_improvement >= args.early_stopping_patience:
            print(f"Early stopping after {epochs_without_improvement} epochs without mIoU improvement.")
            break

    print(f"Done. Best val mIoU: {best_val_miou:.4f}. Output: {run_dir}")
    if args.source_tiles:
        if not math.isfinite(best_val_miou):
            raise RuntimeError('Training ended without a finite validation result')
        (run_dir/'training_complete.json').write_text(json.dumps({
            'best_val_miou':best_val_miou, 'status':'completed',
            'manifest_sha256':(run_dir/'dataset_manifest_sha256.txt').read_text(),
        }, indent=2))


if __name__ == "__main__":
    main()
