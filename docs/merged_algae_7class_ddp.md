# 7-class fresh DDP + overlap validation 학습

## 목적

`Labeling_Data_v2`의 고정 train/validation/test 분할에서 다음 7개 클래스를 학습한다.

```text
river, land, bridge, other, nps, turbid, algae_including_nps_algae
```

데이터 분할은 [Labeling_Data_v2 분할 문서](labeling_data_v2_split.md)를 따른다.

## 초기화

이 run은 이전 프로젝트의 binary/7-class/11-class 체크포인트를 사용하지 않는다.

- Backbone: 외부 사전학습 `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Decoder: 공식 generic `facebook/mask2former-swin-large-ade-semantic`에서 호환되는
  decoder tensor만 초기화
- 새 입력 adapter 및 7-class 출력층: 새로 초기화
- `init_from`: 사용 안 함
- `resume_from`: 최초 시작 시 사용 안 함

즉, 이전 하천 데이터 fine-tuned 모델에서 이어서 학습하는 실험이 아니다.

## DDP와 배치

- GPU: 물리 GPU 0, 1
- 실행: `torch.distributed.run --nproc_per_node=2`
- GPU당 batch: 2
- gradient accumulation: 3
- effective batch: `2 × 2 × 3 = 12`
- 각 rank는 `DistributedSampler`로 서로 다른 train 프레임을 받는다.
- 학습 metric과 confusion matrix는 두 rank에서 합산한다.

## 학습 sampling

- 768×768 random crop
- horizontal flip 0.5
- `class_sampling_mode: within_image`
- `class_sampling_prob: 0.3`

이전 `global 0.5` 방식처럼 선택된 희귀 클래스의 다른 이미지로 현재 샘플 자체를
교체하지 않는다. 원래 프레임 노출 빈도를 유지하면서 해당 프레임 안에서 crop 위치만
클래스 쪽으로 유도한다. 이는 test에서 나타난 `turbid` 과예측을 줄이기 위한 변경이다.

## Validation 및 모델 선택

- 5 epoch마다 전체 validation
- tile 768, stride 512
- 마지막 타일을 영상 끝에 맞춰 zero padding 제거
- 겹친 위치의 semantic score를 중앙 가중 평균한 후 전체 장면 prediction 생성
- 전체 장면 confusion matrix로 GT-supported mIoU 계산
- 두 DDP rank가 서로 다른 원본 장면을 평가하고 confusion matrix를 합산
- 이 overlap validation mIoU로 best checkpoint와 early stopping 결정
- 4회 연속 validation 미개선 시 종료
- 학습 중 test는 열거나 평가하지 않음

## 실행 및 확인

설정 파일:
[`config/labeling_merged_algae_7class_v3_ddp.yaml`](../config/labeling_merged_algae_7class_v3_ddp.yaml)

실행기:
[`tools/run_merged_algae_7class_ddp.py`](../tools/run_merged_algae_7class_ddp.py)

현재 서버에서는 다음 user service로 실행한다.

```bash
systemctl --user status dummdumm-algae7-ddp-v3.service --no-pager
tail -f outputs/labeling_merged_algae_7class_v3_ddp/train.log
cat outputs/labeling_merged_algae_7class_v3_ddp/state.json
cat outputs/labeling_merged_algae_7class_v3_ddp/merged_algae7_ddp_overlap_seed42/progress.json
```

출력 경로:

```text
outputs/labeling_merged_algae_7class_v3_ddp/merged_algae7_ddp_overlap_seed42/
```

최초 실행 전 실제 전체 모델로 DDP forward/backward 46 step을 확인했고, 별도 2-rank
overlap validation smoke test에서 rank 합산 행렬과 Altum+P1 행렬 합계가 일치함을
검증했다.
