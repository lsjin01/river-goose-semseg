# Fusion v3: 데이터 분석 후 재학습 (2026-09-09)

## 분석 범위와 확인 결과

`data/fusion_v2_geo/manifest.json`의 train 539장과 validation 142장,
총 681장을 실제 학습 loader와 동일하게 rasterize해서 집계했다.
이번 분석에서 test 원본 영상/마스크는 열지 않았다. Test의 파일 해시·task·분할
메타데이터만 분할 중복 검증에 사용했다. 분할과 train-only 정규화는 변경하지 않았다.

Manifest SHA256: `c621efd7254d2c37c8b1e5de4a084086ae6fccf14af9ae48e96bfc4991e96826`.

아래 비율은 **전체 원본 annotation-grid의 유효 라벨 픽셀 기준**이다.
실제 학습은 이미지당 crop을 뽑으므로 학습 batch의 픽셀 비율과는 다르다.

| 클래스 | Train 픽셀 비율 | Val 픽셀 비율 | Train 포함 이미지 | Train 독립 그룹 |
|---|---:|---:|---:|---:|
| land | 52.4750% | 40.6890% | 535 | 7 |
| bridge | 0.2969% | 0.0485% | 23 | 6 |
| other | 0.2515% | 0.1535% | 187 | 7 |
| nps | 0.8883% | 1.6520% | 203 | 7 |
| algae0 | 25.8252% | 없음 | 221 | 1 |
| algae1 | 0.3585% | 11.1155% | 6 | 3 |
| algae2 | 16.2597% | 31.6788% | 184 | 5 |
| algae3 | 2.4724% | 14.0344% | 137 | 5 |
| algae4 | 0.0654% | 0.0024% | 2 | 1 |
| turbid | 1.0119% | 0.0027% | 77 | 5 |
| nps_algae | 0.0953% | 0.6232% | 75 | 3 |

독립 그룹은 기존 P1 GPS 연결 성분과 Altum task 단위다. Altum 지역 및
센서 간 동일 장소 여부는 메타데이터가 없어 보증할 수 없다.
Val 전체는 P1 지역 그룹 1개와 Altum task 1개뿐이다. 누수 검사를 통과한
분할이라도 여러 지역에 대한 성능을 안정적으로 대표한다고 볼 수는 없다.

- 라벨 polygon의 영상 범위 이탈 및 rasterize 후 소실 클래스: 0건.
- 미라벨 비율: train 약 0.000164%, val 약 0.000071%.
- 클래스·센서별 대표 영상의 영상/마스크/overlay를 육안 확인했다. 확인한 표본에서
  전반적인 좌표 어긋남은 보이지 않았으나, 다리·수면 등의 일부 라벨 경계는 거칠다.
  이는 전체 라벨의 정확성이나 algae 단계의 물리적 타당성을 검증한 것은 아니다.
- P1과 Altum의 명암·색 표현, 촬영 그룹별 수면 모습이 다르다.
  특히 algae1의 val 픽셀이 train보다 훨씬 큰 비중을 차지한다.
- Altum은 7개 uint16 page를 전부 유지한다. 물리적 밴드 이름은 미확인이다.
  이번에는 정규화·클래스 정의·라벨을 변경하지 않는다.

## 기존 성능과 샘플링 문제

V2 A 최고 epoch 35의 val mIoU는 28.9271%. algae1 0.4768%, algae4 0%,
turbid 0.2177%의 IoU가 병목이다. algae4 예측 면적은 정답의 약 863배,
turbid는 약 131배다. 예측 면적만으로 오분류의 출발 클래스를 알 수 없으므로,
v3에서는 매 검증마다 전체·센서·task별 confusion matrix를 저장한다.

기존 global class-first sampling은 확률 0.7로 클래스를 균등 선택한 뒤
해당 클래스가 있는 이미지로 원래 이미지를 **교체**한다. 학습 원본 수 N=539,
클래스 수 C=11, 클래스 포함 이미지 수 n_c일 때 한 이미지의 epoch당 예상 선택은
`0.3 + (0.7*N/C) * sum(1/n_c for c in image_classes)`이다.
Drop-last 이전 539회 기준, algae4를 가진 task53의 두 이미지는 각각
약 18.58회, 19.21회 선택될 수 있다. 과도한 반복이 오탐의 원인이라는 가설은
아직 검증되지 않았지만, 이를 통제할 명확한 근거는 있다.

## 새 실험: 두 가지 crop 샘플링 비교

| 순서 | 설정 | 이미지 선택 | Crop 선택 |
|---|---|---|---|
| A | `config/fusion_v3/a_within_image.yaml` | 원래 shuffle 순서 유지 | 20%는 해당 이미지 내 클래스/픽셀 선택, 80% 랜덤 |
| B | `config/fusion_v3/b_random_crop.yaml` | 원래 shuffle 순서 유지 | 100% 랜덤 |

두 설정은 run 이름과 class_sampling_prob 외에는 동일하다.
V2 A와 동일한 learned 7-to-3 input fusion을 사용하며 별도 사전학습 가중치에서
각각 새로 시작한다. V2 task 학습 체크포인트를 초기값으로 사용하지 않는다.
7개 page 모두 학습 가능한 입력이다. Loader의 추가 2개 비율 슬롯은 A형 모델에서
사용하지 않는다. P1은 존재하는 RGB 슬롯만 채운다.

GPU 0/1, batch 4, gradient accumulation 3, tile 768, seed 42,
head/adapter LR 4e-5, ViT LR 1e-6, weight decay .02, 최대 100 epoch,
5 epoch마다 전체 val 4,278 tile 평가, 4회 연속 검증 개선 없음이면 종료한다.
이는 4 epoch가 아니라 통상 20 epoch의 patience다. Batch drop-last로
539개 중 536개의 이미지가 각 epoch에서 사용된다. TTA와 CLS aux는 사용하지 않는다.

Primary val mIoU는 기존과 같은 GT-supported 10개 클래스 기준이다.
센서/task별 지표는 각각의 GT-supported class 집합과 함께 기록하므로,
서로 다른 그룹의 mIoU를 단순 평균하거나 직접 동등 비교하지 않는다.
학습 클래스별 픽셀 수와 IoU도 저장한다. Train 지표는 증강 crop 기준이므로
전체 val과 동일한 분포의 측정이 아니다.

분할을 더 쉽게 재배치하거나 test로 설정을 선택하지 않는다. 이 두 실험은
원인 확인용이며 성능 향상을 보장하지 않는다. 이번 큐는 test 평가를 자동 실행하지 않는다.
희귀 클래스의 독립 촬영 장면 부족은 샘플링만으로 해결할 수 없다.

## 보존·재현·실행

V2 서비스는 새 분석/학습 요청에 따라 중지했다. A 완료 결과 및 B의 best/latest는
`outputs/fusion_v2`에 보존했다. 당시 코드/설정은
`outputs/fusion_v2/source_before_v3.tar.gz`에 보관했다.
현재 코드는 v3 변경이 포함되어 있어 v2 큐를 그대로 재개하면 해시 검사가 차단한다.
이전 프로토콜을 복원하려면 보관된 소스를 **별도 작업 디렉터리**에서 사용해야 한다.

분석 산출물(로컬, gitignore):

- `outputs/fusion_v3_analysis/analysis.json`: 분할/클래스/task별 실제 픽셀 통계.
- `outputs/fusion_v3_analysis/image_pixels.csv`, `group_pixels.csv`.
- `outputs/fusion_v3_analysis/{train,val}_{Altum,P1}_overlays.jpg`.

새 실행 산출물은 `outputs/fusion_v3`에 저장한다. 큐는 소스/설정/manifest/분석 결과의
해시를 고정하고 소스 snapshot을 저장한다. 실행 중 이 입력들을 변경하지 않는다.
각 run은 best 1개와 latest를 보관하며, 디스크 여유 15 GiB 미만이면 다음 실행을 중단한다.
재개는 latest를 사용하지만 RNG 전체 상태가 복원되지 않아 bit-identical 재개는 아니다.

```bash
# 데이터 분석 재현: 진행 중 큐가 고정한 분석 파일을 덮어쓰지 않도록 별도 출력 사용
/opt/conda/envs/goose/bin/python tools/analyze_fusion_data.py --output outputs/fusion_v3_analysis_recheck

# 새 큐 실행 또는 latest부터 재개 (이미 서비스 실행 중이면 중복 실행하지 않기)
/opt/conda/envs/goose/bin/python tools/run_fusion_v3_queue.py

# 진행 확인
systemctl --user status dummdumm-fusion-v3-queue.service
tail -f outputs/fusion_v3/a_within_image_train.log
tail -n 10 outputs/fusion_v3/queue_status.jsonl
```

각 run의 `epoch_metrics.csv`, `train_per_class_metrics.csv`,
`val_groups_epoch_XXX.json`을 함께 확인한다. 두 실험 완료 시 validation 비교만 포함한
`outputs/fusion_v3/summary.json`이 작성된다.
