#!/usr/bin/env python3
"""Competition-style fine + coarse mIoU evaluation for DINOv3+M2F checkpoints.

Slim port of the original goose-semseg-challenge evaluation.py that drops
the convnext / vit / DDRNet branches and uses goose_semseg's normalized dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# --- sys.path setup so this script works when run directly ---
_TOOLS = Path(__file__).resolve().parent
_REPO = _TOOLS.parent
for _p in (_TOOLS, _REPO, _REPO / "third_party"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import tqdm  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from inference import (  # noqa: E402
    LoadedCheckpointModel,
    build_model_from_checkpoint,
    run_model_logits,
)
from goose_semseg.data.dataset import GooseSegmentationDataset  # noqa: E402

# -----------------------------
# Competition settings
# -----------------------------
EXCLUDED_FINE_CLASS_IDS = [0, 7, 9, 35, 44, 56, 61, 63]

COARSE_CATEGORIES = [
    "Animal",
    "Construction",
    "Human",
    "Object",
    "Road",
    "Sign",
    "Sky",
    "Terrain",
    "Vegetation",
    "Vehicle",
    "Water",
]
INVALID_COARSE_ID = len(COARSE_CATEGORIES)  # 11

DEFAULT_CLASSNAME_TO_COARSE = {
    "traffic_cone": "Sign",
    "snow": "Terrain",
    "cobble": "Road",
    "obstacle": "Object",
    "leaves": "Vegetation",
    "street_light": "Sign",
    "bikeway": "Road",
    "ego_vehicle": "Vehicle",
    "pedestrian_crossing": "Road",
    "road_block": "Object",
    "road_marking": "Road",
    "car": "Vehicle",
    "bicycle": "Vehicle",
    "person": "Human",
    "bus": "Vehicle",
    "forest": "Vegetation",
    "bush": "Vegetation",
    "moss": "Vegetation",
    "traffic_light": "Sign",
    "motorcycle": "Vehicle",
    "sidewalk": "Road",
    "curb": "Road",
    "asphalt": "Road",
    "gravel": "Terrain",
    "boom_barrier": "Sign",
    "rail_track": "Construction",
    "tree_crown": "Vegetation",
    "tree_trunk": "Vegetation",
    "debris": "Object",
    "crops": "Vegetation",
    "soil": "Terrain",
    "rider": "Human",
    "animal": "Animal",
    "truck": "Vehicle",
    "on_rails": "Vehicle",
    "caravan": "Vehicle",
    "trailer": "Vehicle",
    "building": "Construction",
    "wall": "Construction",
    "rock": "Terrain",
    "fence": "Construction",
    "guard_rail": "Construction",
    "bridge": "Construction",
    "tunnel": "Construction",
    "pole": "Construction",
    "traffic_sign": "Sign",
    "misc_sign": "Sign",
    "barrier_tape": "Sign",
    "kick_scooter": "Vehicle",
    "low_grass": "Vegetation",
    "high_grass": "Vegetation",
    "scenery_vegetation": "Vegetation",
    "sky": "Sky",
    "water": "Water",
    "wire": "Construction",
    "heavy_machinery": "Vehicle",
    "container": "Object",
    "hedge": "Vegetation",
    "barrel": "Object",
    "tree_root": "Vegetation",
}


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def fill_coarse_mapping_from_class_names(class_names, class_to_coarse,
                                         coarse_name_to_id) -> int:
    applied = 0
    for class_id, class_name in enumerate(class_names):
        if class_to_coarse[class_id] != INVALID_COARSE_ID:
            continue

        coarse_name = DEFAULT_CLASSNAME_TO_COARSE.get(
            class_name.strip().lower())
        if coarse_name is None:
            continue

        class_to_coarse[class_id] = coarse_name_to_id[coarse_name]
        applied += 1

    return applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="Path to GOOSE dataset root")
    parser.add_argument("ckpt", type=str, help="Path to checkpoint to load")
    parser.add_argument("--output", "-o", type=str, default="output", help="Path for output")

    # Results
    parser.add_argument(
        "--iou",
        type=str2bool,
        default=True,
        help=
        "Whether to calculate the competition scores or not. [Default True]",
    )
    parser.add_argument(
        "--vis_res",
        type=str2bool,
        default=False,
        help="Whether to visualize the results or not. [Default False]",
    )

    # Data
    parser.add_argument("--n_classes", "-nc", type=int, default=64)
    parser.add_argument(
        "--test_split_name",
        type=str,
        default="val",
        help="Split to evaluate on. Use val for local evaluation.",
    )
    parser.add_argument(
        "--label_mapping_csv",
        type=str,
        default=None,
        help="Path to goose_label_mapping.csv. "
        "If not set, {path}/goose_label_mapping.csv will be used.",
    )

    return parser.parse_args()


def strip_file_prefix(path: str) -> str:
    return path.removeprefix("file://")


def nanmean_tensor(x: torch.Tensor) -> torch.Tensor:
    valid = ~torch.isnan(x)
    if valid.sum() == 0:
        return torch.tensor(float("nan"), dtype=torch.float64)
    return x[valid].mean()


def get_first_existing(row, candidates):
    for key in candidates:
        if key in row and row[key] != "":
            return row[key]
    return None


def resolve_label_mapping_csv(dataset_root: str, csv_path: str) -> str:
    if csv_path is not None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"label_mapping_csv not found: {csv_path}")
        return csv_path

    default_path = os.path.join(dataset_root, "goose_label_mapping.csv")
    if not os.path.exists(default_path):
        raise FileNotFoundError(
            "Could not find goose_label_mapping.csv.\n"
            "Please pass it explicitly with --label_mapping_csv")
    return default_path


def make_iou_tensor(tp: torch.Tensor, union: torch.Tensor) -> torch.Tensor:
    ious = torch.full_like(union, float("nan"), dtype=torch.float64)
    valid = union > 0
    ious[valid] = tp[valid] / union[valid]
    return ious


def load_label_mapping(csv_path: str, n_classes: int):
    """
    Returns:
        class_names: list[str] of length n_classes
        class_to_coarse: torch.LongTensor of length n_classes
            - valid coarse ids: 0~10
            - INVALID_COARSE_ID(=11): ignored in coarse evaluation
    """
    class_names = [str(i) for i in range(n_classes)]
    class_to_coarse = [INVALID_COARSE_ID] * n_classes
    coarse_name_to_id = {
        name: idx
        for idx, name in enumerate(COARSE_CATEGORIES)
    }

    parsed_rows = 0

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            class_id_raw = get_first_existing(
                row, ["class_id", "id", "label_id", "train_id", "label_key"])
            if class_id_raw is None:
                continue

            class_id = int(class_id_raw)
            if class_id < 0 or class_id >= n_classes:
                continue

            class_name_raw = get_first_existing(
                row, ["class_name", "name", "label_name"])
            if class_name_raw is not None:
                class_names[class_id] = class_name_raw.strip()

            coarse_name_raw = get_first_existing(
                row,
                [
                    "category_name",
                    "category",
                    "superclass",
                    "super_class",
                    "coarse_category",
                    "coarse_name",
                ],
            )

            coarse_id_raw = get_first_existing(
                row,
                [
                    "category_id",
                    "coarse_id",
                    "superclass_id",
                    "super_class_id",
                ],
            )

            coarse_id = INVALID_COARSE_ID

            # Prefer category_name if available
            if coarse_name_raw is not None:
                coarse_name = coarse_name_raw.strip()
                if coarse_name.lower() != "void":
                    if coarse_name not in coarse_name_to_id:
                        raise ValueError("Unknown coarse category '{}' in {}\n"
                                         "Expected one of: {}".format(
                                             coarse_name, csv_path,
                                             COARSE_CATEGORIES))
                    coarse_id = coarse_name_to_id[coarse_name]

            # Fallback to category_id if category_name is absent
            elif coarse_id_raw is not None:
                tmp_id = int(coarse_id_raw)
                # support both 0-based and 1-based ids
                if 0 <= tmp_id < len(COARSE_CATEGORIES):
                    coarse_id = tmp_id
                elif 1 <= tmp_id <= len(COARSE_CATEGORIES):
                    coarse_id = tmp_id - 1

            class_to_coarse[class_id] = coarse_id
            parsed_rows += 1

    if parsed_rows == 0:
        raise ValueError(
            "Failed to parse any rows from label mapping CSV: {}\n"
            "Please check the column names.".format(csv_path))

    # If no coarse columns exist in CSV, all entries remain INVALID_COARSE_ID.
    # Fallback to built-in GOOSE class_name -> coarse mapping.
    if any(x == INVALID_COARSE_ID for x in class_to_coarse):
        applied = fill_coarse_mapping_from_class_names(
            class_names=class_names,
            class_to_coarse=class_to_coarse,
            coarse_name_to_id=coarse_name_to_id,
        )
        remaining = sum(x == INVALID_COARSE_ID for x in class_to_coarse)
        print("[INFO] Applied {} fallback coarse mappings from class_name.".
              format(applied))
        if remaining > 0:
            print(
                "[WARNING] {} classes still have no coarse mapping and will be ignored "
                "in coarse evaluation.".format(remaining))

    return class_names, torch.tensor(class_to_coarse, dtype=torch.long)


def write_results_txt_and_json(
    output_path: str,
    config: dict,
    class_names,
    fine_class_ids,
    fine_ious,
    coarse_ious,
    miou_fine,
    miou_coarse,
    miou_composite,
):
    output_dir = Path(output_path)
    lines = [
        "Competition-style Evaluation",
        "=" * 60,
        f"Excluded fine classes: {EXCLUDED_FINE_CLASS_IDS}",
        "",
        f"mIoUfine      : {miou_fine.item()}",
        f"mIoUcoarse    : {miou_coarse.item()}",
        f"mIoUcomposite : {miou_composite.item()}",
        "",
        "[Fine IoU per class]",
        *[
            "{:>2d} ({:<20s}) : {}".format(cls_id, class_names[cls_id],
                                           fine_ious[idx].item())
            for idx, cls_id in enumerate(fine_class_ids)
        ],
        "",
        "[Coarse IoU per category]",
        *[
            "{:<12s} : {}".format(cat_name, coarse_ious[idx].item())
            for idx, cat_name in enumerate(COARSE_CATEGORIES)
        ],
    ]

    with open(output_dir / "results.txt", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
            print(line)

    result_json = {
        "config": config,
        "excluded_fine_class_ids": EXCLUDED_FINE_CLASS_IDS,
        "mIoUfine": float(miou_fine.item()),
        "mIoUcoarse": float(miou_coarse.item()),
        "mIoUcomposite": float(miou_composite.item()),
        "fine_per_class": {
            str(cls_id): {
                "class_name":
                class_names[cls_id],
                "iou":
                None if torch.isnan(fine_ious[idx]) else float(
                    fine_ious[idx].item()),
            }
            for idx, cls_id in enumerate(fine_class_ids)
        },
        "coarse_per_category": {
            COARSE_CATEGORIES[idx]: (None if torch.isnan(coarse_ious[idx]) else
                                     float(coarse_ious[idx].item()))
            for idx in range(len(COARSE_CATEGORIES))
        },
    }

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def visualize(img: torch.Tensor, gt: torch.Tensor, res: torch.Tensor):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img.permute(1, 2, 0).numpy())
    axes[0].axis("off")
    axes[0].set_title("RGB Image")

    axes[1].imshow(gt.numpy())
    axes[1].axis("off")
    axes[1].set_title("Ground Truth")

    axes[2].imshow(res.numpy())
    axes[2].axis("off")
    axes[2].set_title("Inferred")

    plt.tight_layout()
    plt.show()


def update_fine_confusion(conf_mat, gt, pred, n_classes):
    """
    conf_mat: [n_classes, n_classes]
    Row = GT, Col = Pred
    GT pixels of excluded fine classes are ignored.
    """
    gt = gt.view(-1).long()
    pred = pred.view(-1).long()

    excluded = torch.tensor(EXCLUDED_FINE_CLASS_IDS, device=gt.device)
    valid = ((gt >= 0)
             & (gt < n_classes)
             & (pred >= 0)
             & (pred < n_classes)
             & ~torch.isin(gt, excluded))

    gt = gt[valid]
    pred = pred[valid]

    if gt.numel() == 0:
        return

    indices = gt * n_classes + pred
    bincount = torch.bincount(indices, minlength=n_classes * n_classes)
    conf_mat += bincount.reshape(n_classes, n_classes)


def update_coarse_confusion(conf_mat, gt, pred, class_to_coarse, n_classes):
    """
    conf_mat: [11, 12]
      - GT coarse categories: 11 valid categories
      - Pred coarse categories: 11 valid + 1 invalid bucket
    GT pixels whose fine class is excluded from scoring or whose coarse class
    maps to invalid/void are ignored.
    """
    gt = gt.view(-1).long()
    pred = pred.view(-1).long()

    valid = (gt >= 0) & (gt < n_classes) & (pred >= 0) & (pred < n_classes)
    gt = gt[valid]
    pred = pred[valid]

    if gt.numel() == 0:
        return

    excluded = torch.tensor(EXCLUDED_FINE_CLASS_IDS, device=gt.device)
    valid_gt = ~torch.isin(gt, excluded)
    gt = gt[valid_gt]
    pred = pred[valid_gt]

    if gt.numel() == 0:
        return

    gt_coarse = class_to_coarse[gt]
    pred_coarse = class_to_coarse[pred]

    # ignore GT that maps to invalid coarse category
    valid_gt = gt_coarse != INVALID_COARSE_ID
    gt_coarse = gt_coarse[valid_gt]
    pred_coarse = pred_coarse[valid_gt]

    if gt_coarse.numel() == 0:
        return

    num_pred_bins = INVALID_COARSE_ID + 1  # 12
    indices = gt_coarse * num_pred_bins + pred_coarse
    bincount = torch.bincount(indices,
                              minlength=len(COARSE_CATEGORIES) * num_pred_bins)
    conf_mat += bincount.reshape(len(COARSE_CATEGORIES), num_pred_bins)


def compute_fine_ious(conf_mat, n_classes):
    conf_mat = conf_mat.to(torch.float64)

    tp = torch.diag(conf_mat)
    row_sum = conf_mat.sum(dim=1)
    col_sum = conf_mat.sum(dim=0)

    fine_class_ids = [
        i for i in range(n_classes) if i not in EXCLUDED_FINE_CLASS_IDS
    ]
    fine_indices = torch.tensor(fine_class_ids, dtype=torch.long)
    union = row_sum[fine_indices] + col_sum[fine_indices] - tp[fine_indices]
    fine_ious = make_iou_tensor(tp[fine_indices], union)
    miou_fine = nanmean_tensor(fine_ious)

    return fine_class_ids, fine_ious, miou_fine


def compute_coarse_ious(conf_mat):
    """
    conf_mat shape: [11, 12]
    """
    conf_mat = conf_mat.to(torch.float64)

    tp = conf_mat[:, :len(COARSE_CATEGORIES)].diag()
    row_sum = conf_mat.sum(dim=1)
    col_sum = conf_mat[:, :len(COARSE_CATEGORIES)].sum(dim=0)

    union = row_sum + col_sum - tp
    coarse_ious = make_iou_tensor(tp, union)
    miou_coarse = nanmean_tensor(coarse_ious)

    return coarse_ious, miou_coarse


def load_model(ckpt: str, device: torch.device) -> LoadedCheckpointModel:
    bundle = build_model_from_checkpoint(Path(ckpt), device)
    return bundle


def infer_mask(img: torch.Tensor, bundle: LoadedCheckpointModel) -> torch.Tensor:
    # Native GOOSE resolutions (e.g. 1920x1080, 1280x720) are not multiples of
    # the model's stride-32 grid, which breaks Multi-Scale Deformable Attention.
    # Resize to the checkpoint's training resolution for the forward pass, then
    # resize logits back to the original HW for mIoU computation.
    original_hw = img.shape[-2:]
    target_w, target_h = bundle.resize_size
    target_hw = (int(target_h), int(target_w))
    inputs = img.unsqueeze(0)
    if inputs.shape[-2:] != target_hw:
        inputs = F.interpolate(
            inputs, size=target_hw, mode="bilinear", align_corners=False
        )
    semantic_logits = run_model_logits(bundle, inputs)
    if semantic_logits.shape[-2:] != original_hw:
        semantic_logits = F.interpolate(
            semantic_logits.float(),
            size=original_hw,
            mode="bilinear",
            align_corners=False,
        )
    return semantic_logits.argmax(dim=1).squeeze(0)


if __name__ == "__main__":
    opt = parse_args()

    n_classes = opt.n_classes
    calculate_iou = opt.iou
    visualize_res = opt.vis_res
    now = datetime.now()

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load label mapping
    if calculate_iou:
        mapping_csv = resolve_label_mapping_csv(opt.path,
                                                opt.label_mapping_csv)
        class_names, class_to_coarse = load_label_mapping(
            mapping_csv, n_classes)
    else:
        class_names = [str(i) for i in range(n_classes)]
        class_to_coarse = torch.full((n_classes, ),
                                     INVALID_COARSE_ID,
                                     dtype=torch.long)

    # Load model
    ckpt = strip_file_prefix(opt.ckpt)
    bundle = load_model(ckpt, device)
    checkpoint_num_classes = int(
        bundle.checkpoint_args.get("num_classes", bundle.num_classes)
    )
    if checkpoint_num_classes != n_classes:
        print(
            "[WARNING] Overriding --n_classes={} with checkpoint num_classes={}."
            .format(n_classes, checkpoint_num_classes))
        n_classes = checkpoint_num_classes

    # Load data: full-resolution, normalized images via the training dataset
    # class. resize_size=None preserves native resolution; flip_prob=0 and
    # enable_random_crop=False keep the val/test split deterministic.
    validation_dataset = GooseSegmentationDataset(
        root=opt.path,
        split=opt.test_split_name,
        resize_size=None,
        flip_prob=0.0,
        enable_random_crop=False,
    )

    # Competition-style confusion matrices
    fine_conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    coarse_conf_mat = torch.zeros(
        (len(COARSE_CATEGORIES), len(COARSE_CATEGORIES) + 1),
        dtype=torch.int64)

    try:
        print("*** Processing images ***")
        print("*************************")
        pbar = tqdm.tqdm(range(len(validation_dataset)))

        with torch.no_grad():
            for i in pbar:
                img, sem_map = validation_dataset[i]
                img_for_vis = img.clone()

                img = img.to(device)
                mask = infer_mask(img, bundle)

                mask = mask.detach().cpu().long()
                sem_map = sem_map.detach().cpu().long()

                if visualize_res:
                    visualize(img_for_vis.cpu(), sem_map, mask)

                if calculate_iou:
                    update_fine_confusion(fine_conf_mat, sem_map, mask,
                                          n_classes)
                    update_coarse_confusion(coarse_conf_mat, sem_map, mask,
                                            class_to_coarse, n_classes)

                    if (i + 1) % 50 == 0 or (i + 1) == len(validation_dataset):
                        _, _, curr_fine = compute_fine_ious(
                            fine_conf_mat, n_classes)
                        _, curr_coarse = compute_coarse_ious(coarse_conf_mat)
                        curr_comp = 0.5 * curr_fine + 0.5 * curr_coarse
                        pbar.set_postfix_str(
                            f"fine={curr_fine.item():.4f}, coarse={curr_coarse.item():.4f}, comp={curr_comp.item():.4f}"
                        )

    except KeyboardInterrupt:
        print("Interrupted by user, saving results until now.")
    except Exception as e:
        print("An error occurred: {}".format(e))
        raise

    output_path = os.path.join(opt.output, "evaluation",
                               now.strftime("%m-%d-%Y_%H-%M-%S"))
    os.makedirs(output_path, exist_ok=True)

    if calculate_iou:
        fine_class_ids, fine_ious, miou_fine = compute_fine_ious(
            fine_conf_mat, n_classes)
        coarse_ious, miou_coarse = compute_coarse_ious(coarse_conf_mat)
        miou_composite = 0.5 * miou_fine + 0.5 * miou_coarse

        write_results_txt_and_json(
            output_path=output_path,
            config=vars(opt),
            class_names=class_names,
            fine_class_ids=fine_class_ids,
            fine_ious=fine_ious,
            coarse_ious=coarse_ious,
            miou_fine=miou_fine,
            miou_coarse=miou_coarse,
            miou_composite=miou_composite,
        )
