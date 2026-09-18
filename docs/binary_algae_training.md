# Binary algae 재학습

- 양성(1): algae0~4 및 nps_algae. 음성(0): land, bridge, other, nps, turbid.
- ignore=255 유지. 원본 이미지, COCO 라벨 및 분할 manifest는 변경하지 않는다.
- BinaryAlgaeTileDataset이 원본 래스터화 결과를 메모리에서 2-class로 변환한다.
- 기존 공간 그룹 분할 유지: train 539 / validation 142 / test 114프레임.
- 초기화: 외부 DINOv3 ViT-L/16 및 Mask2Former 사전학습. 기존 11-class 학습 체크포인트에서 이어 학습하지 않는다.
- 입력/백본/손실/샘플링/학습률은 CLS-aux 0.05 실험과 동일. 출력만 2-class, CLS-aux도 2-class 존재 여부로 변경한다.
- CLS-aux: BCE, weight=0.05. 분류 0도 실제 음성 클래스이며 ignore가 아니다.
- GPU 0,1 / 총 batch=4 / gradient accumulation=3 / 유효 batch=12 / tile=768 / seed=42.
- 최대 100 epoch, 5 epoch마다 validation, 4회 검증 연속 개선 없으면 조기 종료(약 20 epoch).
- 모델 선택: validation 2-class mIoU. 학습 중 test 평가와 test 기준 튜닝 없음.
- 기존 test 결과를 이미 분석한 후 시작하는 후속 실험이므로, test가 완전히 미공개인 독립 최종 평가라고 주장하지 않는다.
- 기존 11-class 예측을 이진화한 test mIoU 92.46%는 참고값이지 이 모델의 학습 결과가 아니다.

## 실행과 기록

설정: `config/labeling_binary_algae.yaml`

먼저 `tests/test_binary_algae.py` 및 기존 데이터 테스트를 실행하고,
`tools/check_fusion_v2.py --config config/labeling_binary_algae.yaml`로
두 센서·두 GPU의 forward/backward, CLS-aux 출력 2개, ViT gradient, 검증 혼동행렬을 확인한다.
이 smoke test는 train/validation만 사용하며 결과 가중치는 실제 학습에 사용하지 않는다.

`tools/run_binary_algae.py`는 설정/소스/manifest/smoke 결과의 SHA256과 소스 스냅샷을 저장하고 학습한다.
학습 로그는 `outputs/labeling_binary_algae/train.log`, 체크포인트는
`outputs/labeling_binary_algae/binary_aux005_seed42/`에 저장한다.
기존 11-class 실험과 체크포인트는 그대로 보존한다.

```bash
tail -f outputs/labeling_binary_algae/train.log
systemctl --user status dummdumm-binary-algae-aux005.service --no-pager
```
