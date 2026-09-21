# River GOOSE Semantic Segmentation

하천 오염원 영상의 의미론적 분할을 위한 독립 프로젝트입니다. DINOv3 백본과
Mask2Former 교사 모델을 학습하고, SegFormer 학생 모델로 지식 증류(KD), TTA 평가,
추론 비용 측정을 수행할 수 있습니다.

RGB·스펙트럼을 결합하는 5가지 방법의 구조, 입력 채널, 장단점과 실험 해석은
[스펙트럼 결합 방법 A–E 상세 README](docs/spectral_fusion/README.md)에 정리되어 있습니다.

## 프로젝트 구조

```text
.
├── goose_semseg/              # 데이터, 모델, 손실함수, 학습 엔진 패키지
├── third_party/dinov3/        # DINOv3 호환 코드
├── config/                    # 재현 가능한 YAML 실험 설정
├── tools/                     # 변환, 평가, 추론, 시각화 도구
├── train.py                   # Mask2Former 교사 모델 학습
├── train_student_kd.py        # SegFormer 학생 모델 지식 증류
├── eval_tta_segmentation.py   # TTA 정량 평가
├── benchmark_inference_cost.py# 추론 시간/메모리 측정
├── pyproject.toml             # 패키지 및 의존성 정의
└── requirements.txt           # pip 설치용 의존성 목록
```

데이터셋, 체크포인트, 로그, 시각화 결과는 저장소에 포함하지 않습니다. 기본적으로
`data/`와 `outputs/` 아래에 두며 `.gitignore`가 이를 보호합니다.

## 설치

```bash
git clone <repository-url>
cd river-goose-semseg
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

CUDA 기반 MS-Deformable-Attention을 사용할 경우 한 번 빌드합니다.

```bash
cd goose_semseg/models/ops/csrc
python setup.py build install
cd ../../../..
```

다른 서버로 데이터·체크포인트·모델 cache를 옮기는 절차와 실행 전 검사는
[다른 서버 실행 가이드](docs/portability.md)를 참고하십시오. 데이터와 체크포인트는
Git에 포함되지 않으므로 코드만 clone해서는 실제 학습을 시작할 수 없습니다.

현재 워크스테이션에서 검증한 정확한 패키지 버전은 `requirements-validated.txt`,
Conda 환경 구성은 `environment.yml`에 기록되어 있습니다. 완료된 Labeling_Data_v2
Teacher/B1/B0 전달용 묶음의 설명은
[`artifacts/labeling_data_v2_7class_models`](artifacts/labeling_data_v2_7class_models/)에
있습니다.

## 학습 데이터 준비

이 프로젝트가 학습하는 대상은 **하천 오염 데이터**입니다. 코드와 일부 경로에 남아
있는 `goose` 명칭은 기존 데이터 로더와 호환되는 디렉터리 형식을 뜻하며, 실제
GOOSE 데이터셋을 사용해야 한다는 의미가 아닙니다.

### 필수 폴더 구조

사용할 하천 오염 데이터는 반드시 `train`, `val`, `test`로 분리하고, 각 이미지와
정답 마스크를 동일한 split 및 scene 폴더에 배치합니다.

```text
data/goose_cat2/
├── images/
│   ├── train/<scene>/<sample>.png
│   ├── val/<scene>/<sample>.png
│   └── test/<scene>/<sample>.png
└── labels/
    ├── train/<scene>/<sample>_labelids.png
    ├── val/<scene>/<sample>_labelids.png
    └── test/<scene>/<sample>_labelids.png
```

예를 들어 다음 두 파일이 한 쌍입니다.

```text
images/train/03.금강/L03_45140_130_20230802_N06_002333.png
labels/train/03.금강/L03_45140_130_20230802_N06_002333_labelids.png
```

파일 구성 규칙:

- 이미지와 라벨은 모두 PNG 형식을 사용합니다.
- 라벨 파일명은 이미지 stem 뒤에 `_labelids`를 붙여 만듭니다.
- `_windshield_vis`, `_front`, `_camera_left`, `_camera_right`, `_realsense`가 이미지
  stem 끝에 있으면 라벨을 찾을 때 해당 센서 접미사를 제거합니다.
- 라벨은 팔레트/RGB 이미지가 아닌 단일 채널 class-ID 마스크여야 합니다.
- `255`는 학습과 평가에서 제외하는 ignore 값입니다.
- 동일 촬영 장면이 서로 다른 split에 중복되지 않도록 분리해야 합니다.

현재 12개 class ID는 다음과 같습니다.

| ID | 클래스 | ID | 클래스 |
|---:|---|---:|---|
| 0 | background | 6 | 축사 |
| 1 | 밭_논 | 7 | 야적퇴비_가축분뇨 |
| 2 | 잔재물 | 8 | 목장 |
| 3 | 배수로 | 9 | 분뇨개별처리시설 |
| 4 | 비닐하우스 | 10 | 부유쓰레기 |
| 5 | 과수원 | 11 | 연못 |

원천 zip 데이터 변환 예시:

```bash
python tools/convert_to_goose.py --dataset-root data/raw --out data/goose_cat2
```

### Holdout 데이터에서 원래 일반 split 복원

새만금 holdout 데이터에는 일반 train/val과 새만금 전체가 있지만, 일반 test의
한강·낙동강·금강·영산강 샘플은 포함되지 않습니다. 따라서 파일 목록만 옮겨서는
완전한 일반 데이터셋을 만들 수 없습니다. 일반 데이터가 있는 환경에서 manifest와
누락된 test 파일만 묶은 보충 bundle을 먼저 생성합니다.

```bash
python tools/restore_general_from_holdout.py pack \
  --general goose_data \
  --holdout goose_data_holdout_saemangeum \
  --bundle data/general_split_restore_bundle
```

`general_split_restore_bundle`을 holdout 데이터만 있는 환경으로 전송한 후 복원합니다.
기본 `symlink` 모드는 기존 holdout 파일을 복사하지 않으므로 추가 공간을 최소화합니다.

```bash
python tools/restore_general_from_holdout.py restore \
  --holdout /path/to/goose_data_holdout_saemangeum \
  --bundle /path/to/general_split_restore_bundle \
  --output /path/to/goose_data \
  --mode symlink
```

복원된 폴더를 독립적으로 이동해야 하면 `--mode copy`를 사용합니다. 동일 파일시스템에서
심볼릭 링크 없이 공간을 절약하려면 `--mode hardlink`를 사용할 수 있습니다.

### Train/validation/test 로컬 샘플 만들기

원본 데이터에서 각 split의 이미지와 라벨을 한 쌍씩 복사해 빠른 실행 확인용
데이터셋을 만들 수 있습니다. 원본 파일은 변경하지 않습니다.

```bash
python tools/create_sample_dataset.py \
  --source goose_data_cat2 \
  --output data/sample_goose \
  --count 1 \
  --seed 42
```

생성되는 구조:

```text
data/sample_goose/
├── images/{train,val,test}/<scene>/<image>.png
├── labels/{train,val,test}/<scene>/<image>_labelids.png
└── manifest.json
```

`manifest.json`에는 선택된 원본 경로, seed, split별 이미지·라벨 경로가 기록됩니다.
`data/`는 `.gitignore`에 포함되어 있으므로 생성된 샘플과 실제 학습 데이터는
GitHub에 올라가지 않습니다. 다른 사용자는 자신의 하천 오염 데이터에서 아래
명령을 실행해 같은 구조의 샘플을 만들면 됩니다.

교사 모델 1 epoch 학습 확인:

```bash
python train.py \
  --data_path data/sample_goose \
  --output_dir outputs/sample \
  --run_name smoke_teacher \
  --epochs 1 --batch_size 1 --num_workers 0 \
  --resize_width 512 --resize_height 512 \
  --num_classes 12
```

검증 또는 테스트 split 평가는 체크포인트를 지정해 실행합니다.

```bash
python eval_tta_segmentation.py \
  --model_type student \
  --checkpoint outputs/student_kd/best.pt \
  --data_path data/sample_goose \
  --split val --num_classes 12 --batch_size 1 --num_workers 0

python eval_tta_segmentation.py \
  --model_type student \
  --checkpoint outputs/student_kd/best.pt \
  --data_path data/sample_goose \
  --split test --num_classes 12 --batch_size 1 --num_workers 0
```

## 실행

교사 모델 학습:

```bash
python train.py --config config/default.yaml
```

학생 모델 KD 학습:

```bash
python train_student_kd.py \
  --teacher_checkpoint outputs/teacher/best.pt \
  --data_path data/goose_cat2 \
  --student_model segformer_b1 \
  --num_classes 12 \
  --output_dir outputs/student_kd
```

평가와 벤치마크의 전체 옵션은 다음 명령으로 확인할 수 있습니다.

```bash
python eval_tta_segmentation.py --help
python benchmark_inference_cost.py --help
python tools/eval.py --help
```

## 새 라벨링 데이터 규모와 평가 한계

확장 데이터 `Labeling_Data_v2`의 실제 3,985프레임 분할 기준, 고정 관리 그룹,
재현 명령과 manifest hash는
[Labeling_Data_v2 train/validation/test 분할](docs/labeling_data_v2_split.md)에
정리되어 있습니다. 현재 7-class 및 binary v2 실험은 train/val/test
2,790/572/623프레임의 이 분할을 사용합니다.

아래는 기존 대규모 AIHub 데이터셋이 아닌 **새 `Labeling_Data`** 기준입니다.
원본은 총 **795프레임, 17개 episode(task)**이며, Altum 326프레임/5개 task와
P1 469프레임/12개 task로 구성됩니다. 여기서 episode는 라벨링 task 단위이며,
실제 비행 1회나 연속 촬영 세션과 일치한다고 보장하지 않습니다.
Altum TIFF의 7개 밴드는 한 프레임의 채널입니다. Crop/tile 수를 원본 프레임이나
독립 촬영 장면 수로 계산하지 않습니다.

| 분할 | Episode(task) | Altum 프레임 | P1 프레임 | 전체 프레임 | 분할 관리 그룹 |
|---|---:|---:|---:|---:|---:|
| Train | 9 | 216 | 323 | 539 | 7 |
| Validation | 5 | 69 | 73 | 142 | 2 |
| Test | 3 | 41 | 73 | 114 | 2 |
| **합계** | **17** | **326** | **469** | **795** | **11** |

분할 기준은 `data/fusion_v2_geo/manifest.json`입니다. 한 task의 모든 프레임은
같은 split에 배치합니다. P1은 촬영 위치가 1 km 이내로 연결된 task를 날짜와
무관하게 같은 그룹으로 묶습니다. 이에 따라 task9/31/48(221프레임),
task7/8/50/52(73프레임), task10/23(73프레임)은 각각 같은 split에 속합니다.
Altum은 위치 정보가 없어 task 단위로만 분리합니다. 따라서 **관리 그룹 11개가
실제 독립 촬영 지역 11개라는 뜻은 아니며**, Altum 및 센서 간 동일 장소 여부는
확인되지 않았습니다.

### 공정성 해석: 누수 방지와 평가 대표성은 별개

- **누수 방지:** manifest 기준 동일 파일 해시/task의 split 간 중복은 없고,
  확인된 P1 인접 지역도 분리했습니다. 모든 장면의 독립성을 보증하는 것은 아닙니다.
- **평가 대표성:** validation 142프레임이 관리 그룹 2개에 집중되어 있습니다.
  많은 프레임이 여러 하천·촬영 조건을 대표하는 것은 아닙니다.
- **클래스별 한계:** `algae0`는 train에만 있고, val의 `algae4`와 `turbid`는
  각각 1프레임에만 있습니다. Train의 `algae1`은 6프레임이며 해당 클래스 픽셀의
  약 90.8%가 한 프레임에 집중되어 있고, `algae4`는 한 그룹의 2프레임뿐입니다.
- **실험 비교:** 같은 코드/분할/평가 규칙의 실험은 이 검증셋 안에서 비교할 수 있지만,
  한 seed의 작은 mIoU 차이를 새로운 하천 전반의 확실한 우열로 일반화하면 안 됩니다.
  Val/test mIoU는 정답이 있는 고정 10개 세분류를 평균하며, 대분류 CLS 정확도나
  7-class mIoU와 구분합니다.

현재 결과는 **제한된 미지 촬영 그룹에 대한 검증 결과**로 해석합니다.
향후 검증은 test를 보존한 채 train/val 내 촬영 그룹을 바꾸는 교차 검증,
희귀 클래스의 다른 장소·날짜 데이터 보강, 클래스별·그룹별 성능 보고를 권장합니다.
점수를 높이기 위해 프레임 단위 랜덤 분할로 되돌리면 인접 장면 누수 위험이 있습니다.
위 추가 검증과 데이터 보강이 완료됐다는 의미는 아닙니다.
상세 분석은 [Fusion v3 데이터 분석](docs/fusion_v3_data_analysis.md),
동일한 백본 수정 코드의 비교 설정은 [CLS-aux 비교 큐](docs/cls_aux_queue.md)를 참고하십시오.

## 재현성 및 저장 정책

새 라벨링 데이터의 공간 분리 및 다중분광 A~E 재실험은
[Fusion v2 프로토콜](docs/fusion_v2_protocol.md)을 참고하십시오.
실제 사용 split은 `data/fusion_v2_geo`이며, train/val/test는 539/142/114장입니다.
P1은 GPS 기반으로 가까운 촬영 지역을 묶고 Altum은 task를 분리합니다.
`algae0`는 한 지역에만 있어 검증·테스트 성능을 측정할 수 없으며,
새 mIoU는 정답이 존재하는 고정 10개 클래스로 계산합니다. 기존 split의 수치와 직접 비교하지 마십시오.

2026-09-09 데이터 분석 후 재학습은 [Fusion v3 분석·실험 기록](docs/fusion_v3_data_analysis.md)을 참고하십시오.
V2 큐는 중지·보존했으며, 동일 분할에서 이미지 교체 없는 crop 샘플링과 일반 랜덤 crop을 비교합니다.
V3 결과는 `outputs/fusion_v3`에 저장하고, 센서/task별 검증 결과를 기록합니다. 이 큐는 test를 자동 평가하지 않습니다.

새 데이터의 `algae0~4`를 묶는 [11개 세분류 → 7개 대분류 매핑](docs/labeling_data_taxonomy.md)도 제공합니다.
기존 세분류는 유지하며, 새 CLS aux에는 명시적으로 `--cls_aux_taxonomy labeling_data`를 사용합니다.
CLS-aux 재학습 설정은 `config/labeling_coarse_aux.yaml`, 실행 도구는 `tools/run_labeling_coarse_aux.py`입니다.
이 실행에는 DataParallel에서 ViT가 잘못 no_grad로 실행되던 오류 수정도 포함되어 있어,
이전 V3 결과와의 차이를 CLS-aux 단독 효과로 해석하면 안 됩니다. 자세한 내용은 위 문서를 참고하십시오.

7-class 모델을 두 GPU DDP로 fresh 학습하고, padding 없는 overlap validation으로
checkpoint를 선택하는 최신 설정은
[7-class fresh DDP + overlap validation](docs/merged_algae_7class_ddp.md)을 참고하십시오.

동일한 9채널 입력을 SegFormer-B1/B0에 증류하는 방법과 실행 큐는
[7-class multispectral SegFormer KD](docs/merged_algae_7class_kd.md)를 참고하십시오.

여러 강에서 학습한 DINOv3+Mask2Former teacher를 확장 Labeling_Data에 이전하는
후속 실험은 [multi-river teacher fine-tuning](docs/multiriver_labeling_finetune.md)을 참고하십시오.

- 실험 설정은 `config/` 또는 각 실행의 `train_args.json`으로 관리합니다.
- 학습 데이터와 체크포인트는 Git 대신 외부 스토리지나 Git LFS로 관리하십시오.
- 공개 전 데이터 라이선스와 개인정보 포함 여부를 반드시 확인하십시오.
- DINOv3 및 기반 프로젝트의 라이선스/출처는 `LICENSE`와 코드 주석을 따릅니다.
