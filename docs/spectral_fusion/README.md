# RGB·스펙트럼 입력 및 결합 방법 A–E

이 문서는 저장소에 구현된 다섯 가지 fusion(정보 결합) 방법을 설명한다.
일반적인 방법 이름만 설명하는 것이 아니라, 실제 입력 채널·연산·초기화·제약을 코드 기준으로 정리했다.

**2026-09-14 설정 기준:** 확장 데이터의 binary 학습은 **A: 입력 혼합(`input`)**을 사용한다.
같은 확장 데이터·binary 조건에서 B–E의 비교 학습을 완료했다는 뜻은 아니다.
이 문서를 작성하면서 실행 중인 학습 코드·설정은 변경하지 않았다.

## 1. 먼저 쉽게 이해하기

여러 채널을 한 장면에 대한 서로 다른 관측이라고 생각하면 된다.

| 방법 | 쉬운 비유 | 결합 위치 | 설정값 |
|---|---|---|---|
| A. 입력 혼합 | 여러 관측을 먼저 한 장의 영상으로 섞어 보여주기 | DINOv3 이전 | `input` |
| B. 비율 지표 추가 | 원래 관측과 관측 간 차이·비율을 함께 보여주기 | DINOv3 이전 | `indices` |
| C. 나중에 결합 | 두 담당자가 따로 분석한 결과를 마지막에 더하기 | 마지막 feature level | `late` |
| D. 게이트 결합 | 위치마다 스펙트럼의 반영량을 조절하기 | 네 feature level | `gated` |
| E. Cross-attention | 영상 특징이 필요한 스펙트럼 특징을 찾아 참고하기 | 마지막 feature level | `cross_attention` |

여기서 **feature(특징)**는 모델이 영상에서 계산한 숫자 표현이다.
RGB 경로와 스펙트럼 경로를 나누는 C–E도 서로 다른 센서의 사진을 자동으로 정렬하거나 짝짓는 기능은 아니다.

## 2. 실제 데이터는 어떻게 들어오는가?

### 2.1 Altum과 P1은 각각 독립적인 학습 샘플

- **Altum:** TIFF에 저장된 7개 uint16 페이지를 모두 읽는다.
- **P1:** JPEG의 RGB만 읽는다. 추가 스펙트럼 채널은 없으며 0으로 채운다.
- Altum 한 장과 P1 한 장을 같은 장면의 쌍으로 만들어 동시에 넣는 구조가 아니다.
- 두 센서의 샘플이 동일한 모델과 학습 파이프라인을 공유한다.
- 현재 데이터는 연속적인 모든 파장의 스펙트럼이나 수백 밴드의 초분광 데이터라고 확인된 것이 아니다.

**Altum의 각 페이지가 정확히 어떤 파장·물리량인지는 아직 검증되지 않았다.**
코드에 남아 있는 B/G/R/NIR/RedEdge/Thermal 관련 주석은 물리적 밴드 매핑을 검증한 근거가 아니다.
아래의 `page0` 등은 0부터 시작하는 TIFF 페이지 인덱스다.

### 2.2 데이터 로더가 만드는 9채널

타일 크기를 `H=W=768`, 배치 크기를 `B`라고 하면 로더 출력은 `[B, 9, H, W]`다.

| 슬롯 인덱스 | Altum | P1 |
|---|---|---|
| 0–2 | 정규화한 page0–2 | 정규화한 B, G, R |
| 3–6 | 정규화한 page3–6 | 0 |
| 7 | `(page3 − page2) / (page3 + page2 + ε)` | 0 |
| 8 | `(page3 − page4) / (page3 + page4 + ε)` | 0 |

P1은 RGB를 BGR 순서로 바꾸고 8-bit 값을 `×257`한 뒤 센서별 정규화를 적용한다.
슬롯 7·8의 비율은 **정규화하기 전 원본 페이지 값**으로 계산하며 `ε=1e-6`이다.
물리적 밴드 순서가 미확인이므로, 이를 검증된 NDVI·NDRE라고 부르지 않고 **페이지 비율 지표**라고 부른다.

센서별·채널별 정규화 통계는 **train만** 사용한다.

```text
low   = train 표본의 1st percentile
scale = max(99th percentile − 1st percentile, 표준편차, 1)
정규화 값 = asinh((원본 값 − low) / scale)
```

P1의 없는 채널은 정규화 후에도 다시 0으로 고정한다.
입력 타일은 COCO annotation 좌표계에서 추출한다. 원본 JPEG가 annotation보다 크면 해당 원본 영역을 annotation 크기로 맞춘다.
학습 시 무작위 crop·좌우 반전을 사용하고, validation/test에서는 겹치지 않는 타일로 전체 annotation 영역을 평가한다.
가장자리 패딩의 정답은 255로 두어 평가에서 제외한다.

### 2.3 방법마다 사용하는 슬롯이 다르다

| 방법 | 실제 사용 | 비율 지표 2개 사용 |
|---|---|---|
| A | 슬롯 0–6을 한꺼번에 처리 | 아니오 |
| B | 슬롯 0–8을 한꺼번에 처리 | 예 |
| C·D·E | 슬롯 0–2는 DINOv3 경로, 3–6은 스펙트럼 CNN 경로 | 아니오 |

즉, 설정의 `input_channels: 9`가 **모든 방법이 9채널 전체를 사용한다는 뜻은 아니다.**

## 3. A — 입력 혼합: `input`

### 구조

```text
7개 채널 ─┬─ 1×1 Conv: 7 → 3 ───────────────────┐
          └─ 1×1 Conv: 7 → 16 → GELU → Conv: 3 ─┤ 더하기
                                                ↓
                                     3채널 학습 표현
                                                ↓
                                    ImageNet mean/std 적용
                                                ↓
                                      DINOv3 → Mask2Former
```

비선형 경로의 마지막 Conv는 `16 → 3`이다. 두 경로 모두 stride 1의 1×1 Conv라 공간 해상도는 유지한다.

```text
Z = Conv7→3(X) + Conv16→3(GELU(Conv7→16(X)))
```

- 선형 경로는 같은 위치의 여러 채널을 가중합한다.
- 비선형 경로는 관측 값에 따라 달라지는 더 복잡한 혼합을 학습한다.
- 이 단계 자체는 이웃 픽셀을 보는 공간 필터가 아니다. 이후 DINOv3가 공간 정보를 처리한다.
- 출력은 모델 입력용 3채널 표현이며, 실제 RGB 사진이나 0–1 범위의 영상이라는 보장은 없다.

### 초기화와 학습

초기 선형 경로는 슬롯 `2, 1, 0`을 선택하고, 나머지 슬롯의 가중치는 0이다.
비선형 경로의 마지막 Conv도 0으로 초기화한다.
P1에는 이것이 BGR→RGB 재배열이지만, Altum에서는 단지 page2/1/0을 선택하는 초기값이다.

따라서 첫 순간부터 모든 밴드가 예측에 동일하게 기여하는 것은 아니다.
**추가 채널도 연산 입력으로 들어오고, 학습 중 가중치가 갱신되면서 반영될 수 있다.**
입력 어댑터와 DINOv3를 함께 학습한다.

### 장점과 한계

- 장점: 기존 3채널 사전학습 백본을 그대로 활용하고, 별도 스펙트럼 feature branch 없이 구현할 수 있다.
- 한계: 7개 채널을 처음부터 3개로 압축하므로 정보 손실 가능성이 있다.
- 한계: 모든 밴드가 들어간다는 사실만으로 모델이 모든 밴드를 실제로 유용하게 사용한다고 입증되지는 않는다.
- 추가 검증: 밴드를 제거했을 때 성능이 얼마나 달라지는지 별도의 ablation이 필요하다.

**현재 확장 데이터 binary 학습이 이 방식이다.**

## 4. B — 페이지 비율 지표 추가: `indices`

### 구조

```text
원본 7채널 + 페이지 비율 2채널
                ↓
      A와 같은 형태의 입력 어댑터
                9 → 3
                ↓
       DINOv3 → Mask2Former
```

선형 경로는 `9 → 3`, 비선형 경로는 `9 → 16 → 3`이다.
모델이 원본 채널의 값뿐 아니라 두 채널 간 상대적인 차이도 입력으로 받는다.

### 왜 비율을 추가하는가?

두 관측 값의 절대 크기 외에 **상대적인 차이**를 직접 제공하는 설계다.
다만 이 데이터에서 실제 성능을 개선하는지는 비교 실험으로 확인해야 한다.
밴드 의미·정합·센서 보정이 미확인인 상태에서 물리적 녹조 지수라고 해석하면 안 된다.

### 장점과 한계

- 장점: 원본 7개 채널을 버리지 않고 파생 정보 2개를 추가한다.
- 한계: 여전히 최종 입력이 3채널이므로 A와 같은 압축 병목이 있다.
- 한계: 분모가 작거나 채널 값이 부정확하면 비율 정보가 불안정할 수 있다.
- P1에는 비율 지표도 0이며, 실제로 없는 스펙트럼을 생성하는 방법이 아니다.

### 과거 버전과의 차이

초기 6채널 실험에서는 어댑터 내부에서 지표를 계산했다.
현재 `source_tiles` 경로는 로더에서 원본 페이지로 미리 계산한 슬롯 7·8을 사용한다.
따라서 과거 B의 점수를 현재 9채널 B 구현의 성능으로 그대로 인용하면 안 된다.

## 5. C — 나중에 특징 결합: `late`

### 구조

```text
앞 3채널 → 슬롯 2/1/0 재배열 → DINOv3 ── F1, F2, F3, F4 ─┐
                                                        │
추가 4채널 → 작은 CNN → 크기 맞춤 → 1×1 투영 ────────────┤ F4에 더함
                                                        ↓
                                                   Mask2Former
```

여기서 F1–F4는 코드의 feature key `"1"`–`"4"`다.
스펙트럼 CNN은 다음 구조이며, 현재 source-tile 경로에서는 첫 Conv의 stride가 4다.

```text
Conv3×3(4 → 64, stride=4) → GroupNorm → GELU
→ Conv3×3(64 → 64) → GroupNorm → GELU
```

```text
F4_new = F4 + M × Projection(Resize(S, F4의 크기))
```

`S`는 스펙트럼 특징, `Projection`은 64채널을 DINO 특징 차원(현재 1024)으로 바꾸는 1×1 Conv다.
`M`은 스펙트럼 존재 여부로, P1에서는 0이다. F1–F3는 바꾸지 않는다.
현재 개선 구현은 Projection을 0으로 초기화하여 처음에는 추가 경로의 출력이 0이 되도록 한다.

### 장점과 한계

- 장점: 추가 스펙트럼을 백본 입력에서 3채널로 압축하지 않고 별도 특징으로 처리한다.
- 장점: 스펙트럼 CNN은 주변 픽셀까지 함께 처리한다.
- 한계: 단 하나의 feature level에서만 결합한다.
- 한계: 단순 덧셈이라 위치별 신뢰도를 명시적으로 조절하지 않는다.
- 중요: **두 개의 DINOv3를 쓰는 것이 아니다.** 추가 경로는 작은 CNN이다.

## 6. D — 위치별 반영량 학습: `gated`

### 구조

C와 같은 RGB-position 경로와 스펙트럼 CNN을 사용하되, F1–F4 각각에서 결합한다.

```text
RGB-position 특징 Fi ─┬───────────────────────────────────┐
                      └─ 스펙트럼 특징 Si와 연결 → gate ─┤
스펙트럼 특징 Si ───────────── 투영 → gate만큼 반영 ───────┤ 더하기
                                                         ↓
                                                      Fi_new
```

현재 개선 구현의 각 level에서 다음 연산을 한다.

```text
Si     = Resize(S, Fi의 크기)
Gi     = sigmoid(Conv1×1(Concat(Fi, Si)))
Fi_new = Fi + M × Gi × Projection_i(Si)
```

- `Gi`는 `[B, 1, Hi, Wi]` 형태의 **위치별 스칼라 게이트**다.
- 0에 가까우면 해당 위치의 스펙트럼 기여를 줄이고, 1에 가까우면 더 반영한다.
- 채널별 독립 게이트도, attention 행렬도 아니다.
- 현재는 RGB-position 특징과 스펙트럼 특징을 함께 보고 gate를 계산한다.
- 초기 버전은 스펙트럼 특징만 gate 입력으로 사용했다.

### 장점과 한계

- 장점: 서로 다른 위치에서 스펙트럼의 반영량을 다르게 학습할 수 있다.
- 장점: 네 feature level에서 정보를 반영한다.
- 한계: gate가 있다고 자동으로 센서 오류나 반사를 올바르게 감지하는 것은 아니다.
- 한계: C보다 결합 연산이 많고, gate가 한쪽 값으로 치우치면 추가 정보의 활용이 제한될 수 있다.

## 7. E — 필요한 스펙트럼 특징을 찾아 참조: `cross_attention`

### 구조

```text
추가 4채널 → 스펙트럼 CNN → 8×8 pooling → 64개 스펙트럼 token → K, V
                                                                     ↓
앞 3채널 → DINOv3 → F4의 각 위치를 token으로 펼침 → Q → Cross-attention
                                                                     ↓
                                             원래 F4에 더함 → Mask2Former
```

**token**은 여기서 한 위치 또는 한 pooled 영역의 특징 벡터다.

- Query(Q): DINO의 F4 특징.
- Key(K), Value(V): 8×8로 pooling한 스펙트럼 특징.
- Attention head 수: 8.
- 스펙트럼 특징은 64채널에서 현재 DINO 차원인 1024로 투영한다.
- F1–F3는 변경하지 않고 F4만 갱신한다.

```text
A = CrossAttention(Q=RGB_position_tokens, K=spectral_tokens, V=spectral_tokens)
F4_new = F4 + M × α × Reshape(LayerNorm(A))
```

`α`는 학습 가능한 스칼라다. 현재 개선 경로에서는 0으로 초기화한다.
이후 α와 추가 경로가 학습되며 기여가 생길 수 있다.

### 장점과 한계

- 장점: 한 RGB-position 위치가 동일 위치만이 아니라 여러 pooled 스펙트럼 영역을 참고할 수 있다.
- 장점: 단순 덧셈보다 유연한 관계를 표현하는 구조다.
- 한계: pooling으로 세밀한 스펙트럼 공간 정보가 줄어든다.
- 한계: attention 계산이 추가된다. A–E의 실제 FPS·메모리 우열은 동일 조건에서 측정해야 한다.
- 한계: 복잡한 구조가 항상 높은 성능을 보장하지 않는다.
- 중요: 서로 다른 센서 사진의 기하학적 정렬이나 pixel correspondence를 자동으로 보장하지 않는다.

## 8. 스펙트럼이 없는 P1과 CLS-aux는 어떻게 처리하나?

### P1의 없는 정보

A·B에서는 없는 밴드와 지표를 0으로 넣는다. 별도의 센서 ID 임베딩은 없다.
C–E에서는 추가 밴드의 절댓값 합이 0인지 검사해 `M`을 계산하고, P1의 스펙트럼 결합 항을 0으로 만든다.
이 검사는 명시적인 센서 메타데이터가 아니라 **입력 값에 따른 존재 여부 휴리스틱**이다.
또한 C–E는 현재 M을 곱하기 전에 스펙트럼 CNN을 계산하므로, P1에서 그 계산 비용까지 생략되는 것은 아니다.

### CLS-aux와 fusion은 별개

CLS-aux는 최종 segmentation과 별도로, DINO의 CLS token에서 crop에 어떤 클래스가 존재하는지 학습하는 보조 손실이다.
현재 binary 학습에서는 두 클래스의 multi-hot 존재 여부를 BCE, 가중치 0.05로 학습한다.

- A·B: 입력 혼합을 거친 뒤 CLS token을 만들므로 보조 손실이 입력 어댑터에도 전달된다.
- C–E: 코드상 CLS token은 feature fusion 이전의 DINO 출력이다. 따라서 **CLS-aux가 추가 스펙트럼 branch를 직접 감독하지 않는다.**
- C–E의 스펙트럼 branch는 segmentation 손실을 통해 학습한다.

따라서 A–E에서 CLS-aux 가중치를 같게 해도, 보조 손실의 gradient가 흐르는 경로까지 동일한 것은 아니다.

## 9. 과거 비교 결과와 해석 범위

아래는 초기 작은 데이터·이전 분할·11-class·6채널 실험의 **최고 validation mIoU**다.
현재 확장 데이터의 binary 결과나 test 결과가 아니다.

| 실험 | 최고 val mIoU | epoch | 로컬 결과 폴더 |
|---|---:|---:|---|
| A | 62.14% | 70 | `outputs/labeling_data_11cls_multispectral_ce_teacher` |
| B | 61.31% | 64 | `outputs/fusion_b_indices` |
| C | 58.02% | 56 | `outputs/fusion_c_late` |
| D | 58.40% | 72 | `outputs/fusion_d_gated` |
| E | 56.65% | 53 | `outputs/fusion_e_cross_attention` |

수치의 근거는 각 폴더의 `epoch_metrics.csv`, 당시 설정은 `train_args.json`이다.
결과 파일은 대용량 학습 산출물이므로 Git 저장소에는 포함하지 않는다.

이 표만으로 현재 구조의 우열을 확정하면 안 되는 이유:

1. 초기 분할은 같은 task의 프레임이 여러 split에 섞였고, 이후 공간 그룹 분할로 수정했다.
2. 6채널 처리에서 원본 TIFF 7페이지를 보존하는 처리로 바뀌었다.
3. normalization, crop 및 feature fusion 초기화·gate 입력·CNN stride가 바뀌었다.
4. 이후 DataParallel에서 ViT 역전파가 차단되던 문제를 발견하고 수정했다. 코드 버전이 다른 수치는 구분해야 한다.
5. 현재는 3,985개 라벨 보유 프레임의 새 분할이며 분류 문제도 binary로 달라졌다.

또한 “초기 A–E가 모두 실행됐다”와 “수정된 조건에서도 A–E 비교를 모두 완료했다”는 다른 주장이다.
**현재 확장 데이터 binary 조건에서 A–E의 완료된 비교표는 아직 없다.**

## 10. 설정과 후속 비교 방법

현재 설정 파일: [labeling_binary_algae_v2.yaml](../../config/labeling_binary_algae_v2.yaml).

```yaml
input_channels: 9
source_tiles: true
fusion_type: input        # input / indices / late / gated / cross_attention
num_classes: 2
segmentation_taxonomy: binary_algae
enable_cls_aux: true
cls_aux_target_type: fine
cls_aux_weight: 0.05
```

후속 B 실험을 별도 경로로 실행하는 **명령어 예시**는 다음과 같다.
문서 작성으로 이 명령을 실행하거나 큐에 등록하지 않았다. 기존 GPU 작업이 끝난 뒤 사용해야 한다.

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
  /opt/conda/envs/goose/bin/python train.py \
  --config config/labeling_binary_algae_v2.yaml \
  --fusion_type indices \
  --output_dir outputs/binary_fusion_comparison \
  --run_name b_indices_seed42
```

C–E는 `fusion_type`과 고유한 `run_name`을 함께 바꾼다.
현재 단일 A 실험 실행기 `tools/run_binary_algae.py`는 설정을 고정 검사하므로, 이 실행기를 그대로 B–E 큐로 사용할 수는 없다.
정식 비교 큐는 별도 버전으로 구성하고 소스·config·manifest 해시를 보존해야 한다.

공정한 비교를 위해 고정할 항목:

- 동일한 train/val/test 이미지와 그룹, 동일한 train-only normalization.
- 동일한 외부 사전학습 초기화 원칙, seed, crop, batch, gradient accumulation, epoch·LR 정책.
- 동일한 binary 라벨 정의와 ignore 처리.
- 동일한 validation 기준으로 체크포인트 선택, TTA 조건 통일.
- test 결과를 보며 fusion을 선택하거나 threshold를 반복 조정하지 않기.
- 가능하면 여러 seed의 평균·변동, 센서별 IoU, FPS·peak memory도 함께 보고하기.

## 11. 코드와 관련 문서

- [입력 로더: SpectralTileDataset](../../goose_semseg/data/spectral_tiles.py)
- [A·B: SpectralInputAdapter / C–E: SpectralFeatureFusion](../../goose_semseg/models/builder.py)
- [binary 라벨 로더](../../goose_semseg/data/binary_algae.py)
- [학습 및 CLS-aux 손실 경로](../../goose_semseg/engine/trainer.py)
- [현재 확장 데이터 binary 학습 프로토콜](../binary_algae_training_v2.md)
- [이전 Fusion v2 프로토콜](../fusion_v2_protocol.md)
- [DataParallel 역전파 문제 및 수정 기록](../labeling_data_taxonomy.md)

이 문서는 구현 설명이다. 7개 페이지의 물리적 밴드 의미, 특정 fusion의 성능 우위,
학습된 채널 활용도를 별도의 검증 없이 확정하는 문서가 아니다.
