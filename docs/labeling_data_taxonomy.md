# 새 Labeling_Data: 세분류 11개 → 대분류 7개

이는 제공 파일의 공식 supercategory가 아니라, 현재 프로젝트에서 추가한 계층이다.
기존 세분류 라벨·모델 출력 11개는 유지한다. 이전 AIHub 농업계/축산계/하천수면
매핑과 혼용하지 않는다.

| 대분류 ID | 코드 이름 | 설명 | 세분류 이름 (학습 ID) |
|---|---|---|---|
| 0 | land | 육지 | land (0) |
| 1 | bridge | 교량 | bridge (1) |
| 2 | other | 기타 | other (2) |
| 3 | nps | NPS | nps (3) |
| 4 | algae | 녹조 단계 | algae0~algae4 (4~8) |
| 5 | turbid | 탁수 | turbid (9) |
| 6 | nps_algae | NPS·녹조 복합 | nps_algae (10) |

- `algae0`도 단계 구분의 공통 부모에 포함한다. 이것이 녹조 존재/오염 양성을 뜻하지는 않는다.
- `nps_algae`는 어느 구성 성분의 픽셀인지 분해할 근거가 없어 별도 복합 그룹으로 유지한다.
  단일 대분류 체계이며 두 부모에 중복 매핑하지 않는다.
- `other`를 육지나 시설로 단정해 합치지 않는다. NPS의 구체적인 라벨 판정 기준은 별도 확인이 필요하다.
- 새 데이터의 0번은 background가 아니라 land다. 대분류/CLS target에서도 제외하지 않는다.
- 원본 COCO category 11인 `ambiguous`는 현행 11-class 변환에서 제외되었다.
  위 ID는 원본 category ID가 아니라 변환 후 학습 ID다. 무시 픽셀 255는 그대로 무시한다.

## 구현과 사용

`goose_semseg/data/labeling_taxonomy.py`에서 매핑, coarse mask 변환,
11x11 → 7x7 confusion matrix 합산을 제공한다. 원본 파일을 덮어쓰지 않는다.

```python
from goose_semseg.data.labeling_taxonomy import remap_mask, fine_to_coarse_mapping
from goose_semseg.data.coarse_labels import build_batch_cls_aux_targets

coarse_masks = remap_mask(fine_masks)  # fine_masks의 값/파일은 변경되지 않음
targets = build_batch_cls_aux_targets(
    fine_masks, target_type='coarse', num_classes=11, num_coarse=7,
    fine_to_coarse=fine_to_coarse_mapping(),
)
```

CLS aux는 crop 안에 어느 대분류가 존재하는지 나타내는 7차원 multi-hot이다.
픽셀 단위 7-class segmentation head를 추가한 것은 아니다. Main segmentation은
계속 11-class이며 새 계층은 선택적으로 보조 학습에 사용한다.

실행 예시(이번 변경에서 학습을 시작하지는 않음):

```bash
CUDA_VISIBLE_DEVICES=0,1 /opt/conda/envs/goose/bin/python train.py \
  --config config/fusion_v3/b_random_crop.yaml \
  --output_dir outputs/labeling_coarse_aux --run_name coarse7_aux \
  --enable_cls_aux --cls_aux_target_type coarse \
  --cls_aux_taxonomy labeling_data --cls_aux_loss_type bce --cls_aux_weight 0.1
```

새 run 경로를 사용하며, 과거 큐/config/체크포인트는 변경하지 않는다.
Source-tile loader에 과거 `legacy` coarse 매핑을 사용하면 오류로 차단한다.
Manifest 클래스 이름/순서가 11-class 체계와 같은지 검사한다.
Source-tile aux의 `weighted_bce`는 아직 지원하지 않는다. 이미지 존재 빈도로 계산한
가중치가 실제 crop 분포를 대표하지 않기 때문이다.

7-class mIoU와 11-class 출력의 기존 GT-supported 10-class mIoU는 서로 다른
과제의 지표다. 합친 algae 단계 사이의 오분류가 정답으로 바뀌므로, 대분류 점수
상승을 세분류 모델 개선이라고 해석하면 안 된다.

## CLS-aux 재학습 설정 (2026-09-10)

요청에 따른 실행 설정은 `config/labeling_coarse_aux.yaml`이다.
V3 B (`b_random_crop`, best val mIoU 31.0094%)와 같은 분할/정규화/랜덤 crop,
7-band input fusion, seed 42, LR, batch 4, accumulation 3을 사용한다.
추가된 항은 `0.1 * BCEWithLogits(CLS logits, 7-way coarse presence)`이다.
11-class segmentation 손실은 그대로 유지한다. 7개 대분류는 상호 배타적인
픽셀 그룹이지만 한 crop에는 여러 그룹이 있을 수 있어 multi-hot BCE를 쓴다.

외부 DINOv3/Mask2Former 사전학습부터 새로 학습한다. 이전 task-trained best에서
이어 학습하지 않는다. CLS head 초기화는 별도 RNG scope를 사용하여 동일 seed의
segmentation 파라미터 초기화 순서를 바꾸지 않는다.
GPU 0/1, 최대 100 epoch, 전체 val은 5 epoch마다 평가하고 4회 연속 개선 없으면
조기 종료한다. best는 기존과 동일한 GT-supported fine-class mIoU로 선택한다.
TTA 없음, test 자동 평가 없음. CLS binary accuracy는 segmentation mIoU와 다르다.

실행 관리:

```bash
# 중복 실행 금지. 필요하면 같은 실행의 latest부터 재개.
/opt/conda/envs/goose/bin/python tools/run_labeling_coarse_aux.py

systemctl --user status dummdumm-labeling-coarse-aux.service
tail -f outputs/labeling_coarse_aux/train.log
```

결과는 `outputs/labeling_coarse_aux/coarse7_aux`에 저장한다.
`train_args.json`과 `cls_aux_taxonomy.json`에서 활성 설정과 실제 매핑을 확인할 수 있다.
`epoch_metrics.csv`에 train/val 보조 손실 및 binary accuracy를 함께 저장한다.
센서/task별 검증 기록도 유지한다. 완료 후 상위 폴더 `summary.json`에 baseline과의
val mIoU 차이를 기록한다. 원본 데이터/기존 결과는 보존한다.

Runner는 config/source/manifest/사전검증 결과를 해시 고정하고 source snapshot을
보관한다. 시작 후 해당 입력을 수정하면 재개가 차단된다. 각 run의 best 1개와 latest를
유지하며, 시작 시 디스크 15 GiB 이상을 요구한다. 중단 후 재개는 전체 RNG 상태를
복원하지 않으므로 중단 없는 실행과 bit-identical하지는 않다.

### 실행 전 발견한 DataParallel 백본 동결 오류

2026-09-10 실제 모델 CLS-aux backward 검사에서 보조 헤드는 gradient가 있지만
입력 어댑터의 gradient가 0인 것을 확인했다. DINO 어댑터 forward가
`any(p.requires_grad for p in backbone.parameters())`로 no_grad 여부를 정했는데,
DataParallel 복제본은 parameter tensor가 학습 가능해도 `parameters()`가 비어 있다.
GPU 0/1의 소형 재현 모델에서도 단일 모델은 2개, 복제본은 0개로 확인했다.

이 때문에 기존 DataParallel V2/V3 실행에서는 `freeze_backbone: false`여도 ViT
forward가 no_grad였다. 공간 어댑터/decoder는 학습되지만 ViT fine-tuning과
ViT를 통과하는 spectral-input gradient가 차단됐다. 이는 낮은 성능에 기여했을
가능성이 있으나 영향을 정량화하려면 수정 후 CLS-aux OFF 대조 실험이 필요하다.

이번 실행에서 명시적인 freeze 플래그와 input.requires_grad로 판단하도록 수정했다.
Frozen ViT라도 학습 가능한 입력 어댑터까지의 gradient는 유지한다.
따라서 V3 B와 config는 CLS-aux 외 동일하지만 **구현 수정도 함께 포함된 비교**다.
성능 차이를 CLS-aux만의 효과로 보고하지 않는다. 기존 소스 snapshot과 체크포인트는 보존한다.
