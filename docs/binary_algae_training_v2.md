# 확장 Labeling_Data (1) binary 학습

- 원본: `Labeling_Data (1).zip`, 해제 경로 `Labeling_Data_v2`.
- 3,986프레임 / 58 annotation task. 기존 795프레임을 이미 포함하므로 별도로 합치지 않는다.
- 라벨 없는 Altum/task85/IMG_0107.tif 한 장은 원본에 보존하되 학습·평가에서 제외한다. 라벨 부재를 음성으로 간주하지 않는다.
- 양성: algae, algae0~4, nps_algae. 음성: river, land, bridge, other, nps, turbid. ambiguous/미라벨 픽셀=255 ignore.
- 3개 task의 category ID 체계가 달라 이름 기준으로 label_id를 붙인다. 원본 COCO ID와 파일은 변경하지 않는다.
- task 및 동일 이미지 SHA256, GPS 1km 근접 연결 그룹 단위로 train/val/test 약 70/15/15 분할.
- 검증된 분할: train 2,790프레임(Altum 1,154/P1 1,636), val 572(247/325), test 623(237/386).
- task 수는 train/val/test 45/7/6, 관리 그룹 수는 23/6/5. Altum task는 장소가 확인된 독립 지역이라는 의미가 아니다.
- 입력 정규화는 새 train에서만 계산한다. GPS가 없는 Altum task 및 교차 센서의 완전한 장소 독립성은 보장할 수 없다.
- 새 분할이므로 기존 795프레임 test 점수와 직접 비교하지 않는다. 기존 결과를 이미 검토한 후속 실험임을 명시한다.
- DINOv3 ViT-L/16 + Mask2Former, 입력 fusion(7개 원본 밴드; P1은 RGB 외 채널 0), 분할 출력 2개.
- 외부 사전학습부터 새로 시작. 기존 11-class fine-tuned 체크포인트를 초기값으로 사용하지 않는다.
- CLS-aux: 2-class 존재 여부, BCE weight 0.05. GPU 0,1, 총 batch=4, accumulation=3, tile=768, seed=42.
- LR=4e-5, backbone LR=1e-6, 최대100 epoch. 5 epoch마다 validation, 검증4회 연속 개선 없으면 종료.
- validation 2-class mIoU로 모델 선택. 학습 중 test 추론/튜닝 없음.

설정: `config/labeling_binary_algae_v2.yaml`. 실행: `tools/run_binary_algae.py`.
실행기는 분할/설정/소스/검증 SHA256 및 소스 스냅샷을 저장한다.

```bash
tail -f outputs/labeling_binary_algae_v2/train.log
systemctl --user status dummdumm-binary-algae-v2.service --no-pager
```
