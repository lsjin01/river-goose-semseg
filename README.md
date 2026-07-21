# River GOOSE Semantic Segmentation

하천 오염원 영상의 의미론적 분할을 위한 독립 프로젝트입니다. DINOv3 백본과
Mask2Former 교사 모델을 학습하고, SegFormer 학생 모델로 지식 증류(KD), TTA 평가,
추론 비용 측정을 수행할 수 있습니다.

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

## 재현성 및 저장 정책

- 실험 설정은 `config/` 또는 각 실행의 `train_args.json`으로 관리합니다.
- 학습 데이터와 체크포인트는 Git 대신 외부 스토리지나 Git LFS로 관리하십시오.
- 공개 전 데이터 라이선스와 개인정보 포함 여부를 반드시 확인하십시오.
- DINOv3 및 기반 프로젝트의 라이선스/출처는 `LICENSE`와 코드 주석을 따릅니다.
