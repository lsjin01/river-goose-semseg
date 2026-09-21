# 다른 서버에서 실행하기

코드는 독립 저장소로 실행할 수 있지만 데이터, 모델 체크포인트, Hugging Face
가중치는 크기와 라이선스 때문에 Git에 포함하지 않는다. 따라서 새 서버에서는 아래
외부 자산을 별도로 준비해야 한다.

## 1. 검증된 환경

현재 서버에서 검증한 조합은 Python 3.9, PyTorch 2.8.0+cu128, torchvision
0.23.0+cu128, transformers 4.57.6, timm 1.0.25이다. NVIDIA driver는 설치한
PyTorch CUDA runtime을 지원해야 하며, CUDA 확장을 빌드하려면 호환되는 `nvcc`가
필요하다.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .

cd goose_semseg/models/ops/csrc
python setup.py build install
cd ../../../..
```

## 2. 외부 모델 캐시

인터넷이 되는 준비 환경에서 필요한 공식 가중치를 캐시한다.

```bash
python -m huggingface_hub.commands.huggingface_cli download \
  facebook/dinov3-vitl16-pretrain-lvd1689m
python -m huggingface_hub.commands.huggingface_cli download \
  facebook/mask2former-swin-large-ade-semantic
python -m huggingface_hub.commands.huggingface_cli download \
  nvidia/segformer-b0-finetuned-ade-512-512
python -m huggingface_hub.commands.huggingface_cli download \
  nvidia/segformer-b1-finetuned-ade-512-512
```

오프라인 서버에는 Hugging Face cache 전체를 같은 사용자 cache 위치로 복사한다.

## 3. 데이터와 체크포인트

다음 항목은 GitHub 저장소에 들어 있지 않으므로 별도로 복사한다.

- 원본 `Labeling_Data_v2/`
- 준비된 `data/labeling_merged_algae_7class_v2_geo/`
- 여러 강 teacher `best.pt` (약 4 GB)

원본 데이터 위치가 기존 manifest와 달라도 다음 환경변수로 지정할 수 있다.

```bash
export RIVER_SEMSEG_SOURCE_ROOT=/datasets/Labeling_Data_v2
```

## 4. 실행 전 검사

```bash
python tools/check_portability.py \
  --config config/labeling_merged_algae_7class_multiriver_ft_gpu1.yaml \
  --init-from /checkpoints/multiriver_teacher_best.pt
```

모든 항목이 `ok: true`일 때 실제 학습을 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config config/labeling_merged_algae_7class_multiriver_ft_gpu1.yaml \
  --init_from /checkpoints/multiriver_teacher_best.pt \
  --output_dir /experiments/labeling_teacher
```

`systemd-run`은 장시간 작업을 유지하기 위해 현재 서버에서 사용한 운영 방식일 뿐이며
필수 의존성이 아니다. 다른 서버에서는 위의 직접 실행, Slurm, Docker 또는 해당
환경의 작업 스케줄러를 사용해도 학습 코드와 결과는 같다.
