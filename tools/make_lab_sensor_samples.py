#!/usr/bin/env python3
"""Create train-only P1/Altum sample panels for a lab meeting."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/labeling_binary_v2_geo/manifest.json"
OUTPUT = ROOT / "docs/assets/lab_meeting_sensor_samples"

# Chosen from train only. Together they expose common and rare source classes,
# while avoiding a contact sheet dominated by near-identical consecutive frames.
SELECTED = {
    "P1": [
        "P1/Images/task53/DJI_20260708180808_0043.jpg",
        "P1/Images/task55/DJI_20260709074915_0032.jpg",
        "P1/Images/task29/DJI_20260701154505_0028.jpg",
        "P1/Images/task23/DJI_20260701141035_0044.jpg",
        "P1/Images/task89/DJI_20260727160748_0053.jpg",
    ],
    "Altum": [
        "Altum/Images/task98/IMG_1066.tif",
        "Altum/Images/task35/IMG_0024.tif",
        "Altum/Images/task27/IMG_0142.tif",
        "Altum/Images/task94/IMG_0378.tif",
    ],
}

COLORS = {
    "river": (45, 105, 210),
    "land": (140, 105, 65),
    "bridge": (135, 135, 145),
    "other": (235, 145, 35),
    "nps": (215, 55, 145),
    "turbid": (145, 80, 205),
    "algae": (25, 115, 60),
    "algae0": (185, 225, 175),
    "algae1": (130, 215, 75),
    "algae2": (45, 185, 85),
    "algae3": (0, 135, 105),
    "algae4": (0, 85, 65),
    "nps_algae": (35, 205, 215),
    "ambiguous": (245, 220, 70),
}


def stretch(array):
    """Robust 16-bit page stretch for presentation, not model preprocessing."""
    array = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(array, (1, 99))
    scaled = np.clip((array - lo) / max(float(hi - lo), 1.0), 0, 1)
    return np.uint8(np.sqrt(scaled) * 255)


def source_preview(source_root, record, width=960):
    path = source_root / record["source_image"]
    with Image.open(path) as image:
        if record["sensor"] == "P1":
            rgb = image.convert("RGB")
            title = "P1 source RGB"
            bands = None
        else:
            pages = []
            for index in range(7):
                image.seek(index)
                pages.append(stretch(np.asarray(image)))
            # A display candidate only: page2/page1/page0 -> R/G/B.
            rgb = Image.fromarray(np.stack((pages[2], pages[1], pages[0]), axis=-1))
            rgb = ImageEnhance.Color(rgb).enhance(1.12)
            rgb = ImageEnhance.Contrast(rgb).enhance(1.08)
            title = "Altum contrast composite: pages 2/1/0 -> R/G/B (unverified)"
            bands = pages
    height = max(1, round(width * record["height"] / record["width"]))
    return np.asarray(rgb.resize((width, height), Image.Resampling.LANCZOS)), title, bands


def display_mask(record, size):
    width, height = size
    sx, sy = width / record["width"], height / record["height"]
    mask = Image.new("RGB", size, (0, 0, 0))
    valid = Image.new("L", size, 0)
    mask_draw, valid_draw = ImageDraw.Draw(mask), ImageDraw.Draw(valid)
    # Same large-to-small ordering as the training rasterizer.
    for annotation in sorted(record["annotations"], key=lambda a: a.get("area", 0), reverse=True):
        name = annotation.get("source_category_name", "ambiguous")
        color = COLORS.get(name, COLORS["ambiguous"])
        for polygon in annotation.get("segmentation", []):
            if len(polygon) < 6:
                continue
            points = [(polygon[i] * sx, polygon[i + 1] * sy) for i in range(0, len(polygon), 2)]
            mask_draw.polygon(points, fill=color)
            valid_draw.polygon(points, fill=255)
    return np.asarray(mask), np.asarray(valid) > 0


def frame_panel(source_root, record):
    rgb, source_title, _ = source_preview(source_root, record)
    colors, valid = display_mask(record, (rgb.shape[1], rgb.shape[0]))
    overlay = rgb.copy()
    overlay[valid] = (0.55 * rgb[valid] + 0.45 * colors[valid]).astype(np.uint8)
    classes = sorted({a.get("source_category_name", "ambiguous") for a in record["annotations"]})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.7))
    for ax, image, title in zip(
        axes,
        (rgb, overlay, colors),
        (f"Image | {source_title}", "Overlay | Image + segmentation", "GT | Segmentation"),
    ):
        ax.imshow(image)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle(
        f'{record["sensor"]}/{record["task"]}/{Path(record["source_image"]).name}\n'
        f'Classes ({len(classes)}): {", ".join(classes)}',
        fontsize=13,
    )
    handles = [Patch(color=np.array(COLORS[name]) / 255, label=name) for name in classes]
    handles.append(Patch(color=(0, 0, 0), label="unlabeled / ignore"))
    fig.legend(handles=handles, loc="lower center", ncol=min(7, len(handles)), frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.13, wspace=0.025)
    return fig, rgb, colors, overlay, classes


def contact_sheet(rows, sensor):
    fig, axes = plt.subplots(len(rows), 3, figsize=(21, 4.25 * len(rows)))
    for row, (record, rgb, _colors, overlay, classes) in enumerate(rows):
        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(overlay)
        axes[row, 2].imshow(_colors)
        axes[row, 0].set_title(
            "Image | P1 source RGB"
            if sensor == "P1"
            else "Image | Altum pages 2/1/0 display composite (unverified)",
            fontsize=10,
        )
        axes[row, 1].set_title(
            f'Overlay | {record["task"]}/{Path(record["source_image"]).name}\n'
            f'{len(classes)} classes: {", ".join(classes)}',
            fontsize=9,
        )
        axes[row, 2].set_title("GT | Original multiclass segmentation", fontsize=10)
        for column in range(3):
            axes[row, column].axis("off")
    present = sorted({name for _, _, _, _, classes in rows for name in classes})
    handles = [Patch(color=np.array(COLORS[name]) / 255, label=name) for name in present]
    handles.append(Patch(color=(0, 0, 0), label="unlabeled / ignore"))
    fig.legend(
        handles=handles,
        loc="lower center", ncol=7, frameon=False,
    )
    subtitle = (
        "All examples are TRAIN frames; test frames were not opened."
        if sensor == "P1"
        else "TRAIN frames only. Display composite is for visualization; TIFF page identities remain unverified."
    )
    fig.suptitle(f"{sensor} diverse multiclass examples\n{subtitle}", fontsize=15)
    # Reserve enough room for the two-line heading so it never covers row 1.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.052, hspace=0.25, wspace=0.018)
    return fig


def altum_band_overview(source_root, record):
    path = source_root / record["source_image"]
    with Image.open(path) as image:
        pages = []
        page_labels = []
        for index in range(7):
            image.seek(index)
            raw = np.asarray(image)
            pages.append(stretch(raw))
            nodata_fraction = float(np.mean(raw == 65534))
            if nodata_fraction > 0.99:
                page_labels.append(f"TIFF page {index} | NoData-like: {nodata_fraction * 100:.2f}% = 65534")
            else:
                page_labels.append(f"TIFF page {index} | independent 1-99% stretch")
    composite = np.stack((pages[2], pages[1], pages[0]), axis=-1)
    fig, axes = plt.subplots(2, 4, figsize=(16, 10))
    axes[0, 0].imshow(composite)
    axes[0, 0].set_title("Display composite: page2/page1/page0 -> R/G/B")
    for index, page in enumerate(pages):
        ax = axes.flat[index + 1]
        ax.imshow(page, cmap="gray", vmin=0, vmax=255)
        ax.set_title(page_labels[index])
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(
        f'Altum seven-page overview | {record["task"]}/{path.name}\n'
        "Display processing only; physical wavelength/page mapping has not been verified",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def altum_band_sample_sheet(source_root, records, page, band_name, wavelength, cmap_name):
    """Show one physical-band working interpretation across several train scenes."""
    fig, axes = plt.subplots(len(records), 3, figsize=(21, 4.25 * len(records)))
    all_classes = set()
    for row, record in enumerate(records):
        rgb, _source_title, _ = source_preview(source_root, record)
        size = (rgb.shape[1], rgb.shape[0])
        with Image.open(source_root / record["source_image"]) as image:
            image.seek(page)
            raw = np.asarray(image, dtype=np.float32)
        lo, hi = (float(value) for value in np.percentile(raw, (1, 99)))
        normalized = np.clip((raw - lo) / max(hi - lo, 1.0), 0, 1)
        normalized = np.asarray(
            Image.fromarray(np.float32(normalized)).resize(size, Image.Resampling.BILINEAR)
        )
        grayscale = np.uint8(normalized * 255)
        false_color = np.uint8(plt.get_cmap(cmap_name)(normalized)[..., :3] * 255)
        gt_colors, valid = display_mask(record, size)
        overlay = false_color.copy()
        overlay[valid] = (0.72 * false_color[valid] + 0.28 * gt_colors[valid]).astype(np.uint8)
        classes = sorted({a.get("source_category_name", "ambiguous") for a in record["annotations"]})
        all_classes.update(classes)

        axes[row, 0].imshow(rgb)
        axes[row, 1].imshow(grayscale, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].imshow(overlay)
        axes[row, 0].set_title("Reference | pages 2/1/0 display composite", fontsize=10)
        axes[row, 1].set_title(
            f"{band_name} grayscale | raw p01={lo:.0f}, p99={hi:.0f}", fontsize=10
        )
        axes[row, 2].set_title(
            f'False color + GT | {record["task"]}/{Path(record["source_image"]).name}\n'
            f'{len(classes)} classes: {", ".join(classes)}',
            fontsize=9,
        )
        for column in range(3):
            axes[row, column].axis("off")

    handles = [Patch(color=np.array(COLORS[name]) / 255, label=name) for name in sorted(all_classes)]
    handles.append(Patch(color=(0, 0, 0), label="unlabeled / ignore"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False)
    fig.suptitle(
        f"Altum {band_name} working visualization | page{page}, approximately {wavelength}\n"
        f"{cmap_name} false color: dark=low, bright=high | Each TRAIN frame is stretched independently",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.052, hspace=0.25, wspace=0.018)
    return fig


def main():
    manifest = json.loads(MANIFEST.read_text())
    source_root = Path(manifest["source"])
    if not source_root.is_absolute():
        source_root = ROOT / source_root
    records = {record["source_image"]: record for record in manifest["records"]}
    OUTPUT.mkdir(parents=True, exist_ok=True)

    summaries = {}
    rendered = {}
    for sensor, paths in SELECTED.items():
        rows = []
        summaries[sensor] = []
        for number, path in enumerate(paths, 1):
            record = records[path]
            assert record["sensor"] == sensor and record["split"] == "train"
            fig, rgb, colors, overlay, classes = frame_panel(source_root, record)
            filename = f"{sensor.lower()}_sample_{number:02d}.png"
            fig.savefig(OUTPUT / filename, dpi=150)
            plt.close(fig)
            rows.append((record, rgb, colors, overlay, classes))
            summaries[sensor].append({
                "file": filename,
                "source_image": path,
                "task": record["task"],
                "classes": classes,
                "split": record["split"],
            })
        fig = contact_sheet(rows, sensor)
        sheet = f"{sensor.lower()}_diverse_samples.png"
        fig.savefig(OUTPUT / sheet, dpi=150)
        plt.close(fig)
        rendered[sensor] = sheet

    altum_record = records[SELECTED["Altum"][0]]
    fig = altum_band_overview(source_root, altum_record)
    fig.savefig(OUTPUT / "altum_seven_page_overview.png", dpi=150)
    plt.close(fig)

    # Keep the band-specific presentation pages compact: two complementary scenes.
    # task98 covers several algae stages; task35 includes bridge/NPS/nps_algae.
    altum_records = [records[path] for path in SELECTED["Altum"][:2]]
    band_figures = [
        (3, "NIR", "840-842 nm", "viridis", "altum_nir_samples.png"),
        (4, "Red Edge", "717 nm", "magma", "altum_red_edge_samples.png"),
        (5, "Thermal/LWIR", "11 um", "inferno", "altum_thermal_lwir_samples.png"),
    ]
    for page, band_name, wavelength, cmap_name, filename in band_figures:
        fig = altum_band_sample_sheet(
            source_root, altum_records, page, band_name, wavelength, cmap_name
        )
        fig.savefig(OUTPUT / filename, dpi=150)
        plt.close(fig)

    lines = [
        "# 랩미팅 센서 샘플과 밴드 설명",
        "",
        "모든 예시는 **train split**에서만 선택했으며 예약된 test 이미지는 열지 않았다.",
        "GT에는 binary 통합 전의 원본 다중 클래스 이름을 표시했다.",
        "GT의 검은색은 semantic class가 아니라 `unlabeled/ignore` 영역이다.",
        "",
        "## P1",
        "",
        "P1은 일반 RGB 영상이다. Red는 적색, Green은 녹색, Blue는 청색 가시광 세기이며,",
        "P1 샘플에는 별도의 NIR·Red Edge·Thermal 측정값이 없다.",
        "현재 공통 로더에서는 P1을 B/G/R 순서의 슬롯 0/1/2에 넣고 나머지 센서 슬롯을 0으로 채운다.",
        "",
        "![P1 diverse samples](p1_diverse_samples.png)",
        "",
        "## Altum",
        "",
        "표시 합성은 각 페이지를 개별 percentile stretch한 뒤 TIFF page2/1/0을 R/G/B에 배치했다.",
        "발표용 표시일 뿐 모델의 전처리 방식은 아니며, 현재 TIFF에는 밴드 이름 메타데이터가 남아 있지 않다.",
        "프로젝트 코드와 공식 Altum imager 순서가 일치하므로 아래 순서를 working mapping으로 사용한다.",
        "최종 논문 전에는 원본 TIFF 생성 명세 또는 센서 원본 파일로 다시 확인해야 한다.",
        "",
        "### Altum 페이지별 의미",
        "",
        "| TIFF page | 프로젝트상 밴드 | 대표 중심 파장 | 측정 의미 | 녹조 분석에서 기대하는 정보 |",
        "|---:|---|---:|---|---|",
        "| 0 | Blue | 약 475 nm | 청색 가시광 반사 | 물·대기 산란과 탁도·부유물 변화; 수중 감쇠 영향도 큼 |",
        "| 1 | Green | 약 560 nm | 녹색 가시광 반사 | 녹색 조류·식생과 물의 가시적 밝기 차이 |",
        "| 2 | Red | 약 668 nm | 적색 가시광 반사 | 엽록소 흡수 관련 대비; NIR과 조합 가능 |",
        "| 3 | NIR | 약 840–842 nm | 근적외선 반사 | 물은 강하게 흡수하고 수면 위 식생·두꺼운 부유 녹조는 상대적으로 밝을 수 있음 |",
        "| 4 | Red Edge | 약 717 nm | Red와 NIR 사이의 반사 전이 | 엽록소·생체량 변화에 민감할 가능성 |",
        "| 5 | Thermal/LWIR | 약 11 μm | 표면 열복사 | 수온·표면 온도의 간접 정보; 녹조를 직접 측정하는 값은 아님 |",
        "| 6 | NoData형 page | 해당 없음 | 거의 전 픽셀 `65534` | 장면 정보가 없으므로 센서 밴드로 해석하지 않음 |",
        "",
        "프로젝트 매핑 근거: [`prepare_labeling_data.py`](../../../tools/prepare_labeling_data.py)와",
        "[`builder.py`](../../../goose_semseg/models/builder.py). 공식 사양:",
        "[MicaSense Altum Integration Guide](https://support.micasense.com/hc/en-us/articles/360010025413-Altum-Integration-Guide).",
        "Thermal 원시값은 보정식과 단위 확인 없이 섭씨 온도로 읽으면 안 된다.",
        "수중 조류는 물의 NIR 흡수·수면 반사·sun glint 영향을 받으므로 NIR/Red Edge가 항상 녹조 농도와 비례한다고 가정하지 않는다.",
        "",
        "![Altum diverse samples](altum_diverse_samples.png)",
        "",
        "## NIR·Red Edge·Thermal 개별 시각화",
        "",
        "각 밴드는 서로 다른 train 장면 2개만 제시한다. 각 행은",
        "`RGB 유사 합성 | 밴드 grayscale | 밴드 false color + 원본 GT` 순서다.",
        "false color에서는 어두운 색이 해당 프레임 내 낮은 값, 밝은 색이 높은 값이다.",
        "각 프레임을 1–99 percentile로 독립 변환했기 때문에 장면 사이의 색을 절대 센서값처럼 비교하면 안 된다.",
        "GT 색은 위치 해석을 돕기 위한 반투명 표시이며 false-color 스케일과 관계없다.",
        "",
        "### NIR — page3",
        "",
        "![Altum NIR](altum_nir_samples.png)",
        "",
        "### Red Edge — page4",
        "",
        "![Altum Red Edge](altum_red_edge_samples.png)",
        "",
        "### Thermal/LWIR — page5",
        "",
        "![Altum Thermal LWIR](altum_thermal_lwir_samples.png)",
        "",
        "## Altum 7페이지 실제 값 비교",
        "",
        "page0~5에는 서로 다른 장면 신호가 보이지만 page6은 NoData형이다.",
        "각 페이지를 독립적으로 명암 확장했으므로 페이지 사이의 절대 밝기를 비교하면 안 된다.",
        "",
        "![Altum page overview](altum_seven_page_overview.png)",
        "",
        "## 선택한 원본 프레임",
        "",
        "| Sensor | Task | Source | Original classes |",
        "|---|---|---|---|",
    ]
    for sensor in ("P1", "Altum"):
        for item in summaries[sensor]:
            lines.append(
                f'| {sensor} | {item["task"]} | `{item["source_image"]}` | '
                f'{", ".join(item["classes"])} |'
            )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n")
    (OUTPUT / "selection.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps({"output": str(OUTPUT), "rendered": rendered}, indent=2))


if __name__ == "__main__":
    main()
