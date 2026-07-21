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

## 데이터 준비

GOOSE 형식은 다음 구조를 사용합니다.

```text
data/goose_cat2/
├── images/{train,val,test}/<scene>/*_windshield_vis.png
└── labels/{train,val,test}/<scene>/*_labelids.png
```

원천 zip 데이터 변환 예시:

```bash
python tools/convert_to_goose.py --dataset-root data/raw --out data/goose_cat2
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
