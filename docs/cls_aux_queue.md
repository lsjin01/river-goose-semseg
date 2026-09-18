# CLS-aux 가중치 비교 큐 (2026-09-10)

현재 실행 중인 `coarse7_aux`를 재시작하거나 중단하지 않고, 완료 후 세 실험을
순차 실행한다. 동일한 **DataParallel 백본 역전파 수정 코드**로 AUX OFF 대조군을
새로 학습하므로, 기존 V3 31.01%와 구분한다.

| 순서 | 실험 | CLS-aux 가중치 | 출력 폴더 |
|---|---|---:|---|
| 1 | 현재 실행 유지 | 0.1 | `outputs/labeling_coarse_aux/coarse7_aux` |
| 2 | 수정 후 AUX OFF 대조군 | 0 | `outputs/cls_aux_queue/a_aux_off` |
| 3 | 약한 보조 손실 | 0.05 | `outputs/cls_aux_queue/b_aux_005` |
| 4 | 강한 보조 손실 | 0.2 | `outputs/cls_aux_queue/c_aux_020` |

모든 실험은 11-class segmentation, 선택적 7-class coarse CLS presence BCE,
동일한 source manifest/정규화/랜덤 crop/seed 42, GPU 0/1, batch 4,
accumulation 3, tile 768, head LR 4e-5 / ViT LR 1e-6을 사용한다.
각각 외부 사전학습부터 시작하며, 앞 실험의 checkpoint를 다음 실험에 전달하지 않는다.
최대 100 epoch, 5 epoch마다 val, 4회 연속 검증 개선 없음이면 조기 종료한다.
Test와 TTA는 사용하지 않는다. 클래스별 성능과 동일 그룹의 mIoU를 함께 비교한다.

현재 0.1 실행이 실패/중단되면 뒤의 큐도 중단한다. 실행은 선행 runner의 파일 잠금이
해제될 때까지 GPU를 사용하지 않고 기다린다. 잠금 해제 후 완료 marker와 manifest를
검증해야 다음 실험으로 넘어간다. 중복 큐는 잠금으로 차단한다.
실행 중인 소스·설정은 수정하지 않는다. 후속 설정만 별도 `config/cls_aux_queue`에 둔다.
각 실행은 best 1개/latest를 보존한다. 시작 전 여유 공간 15 GiB를 검사한다.
기존 데이터와 checkpoint는 삭제하지 않는다.

```bash
# 읽기 전용 사전 검증
/opt/conda/envs/goose/bin/python tools/run_cls_aux_queue.py --check-only

# 실행 또는 이 큐의 latest부터 재개 (서비스 실행 중 중복 실행 금지)
/opt/conda/envs/goose/bin/python tools/run_cls_aux_queue.py

systemctl --user status dummdumm-cls-aux-queue.service
cat outputs/cls_aux_queue/queue_state.json
tail -n 10 outputs/cls_aux_queue/queue_status.jsonl
tail -f outputs/cls_aux_queue/a_aux_off_train.log
```

`outputs/cls_aux_queue/plan.json`에 순서가, `queue_inputs.json`과
`source_snapshot.tar.gz`에 실행 입력이 기록된다. 완료 후 `summary.json`은
4개 실험의 best fine-class val mIoU, 클래스별 지표, 수정 후 AUX OFF 대비 차이를
모은다. Seed 1개씩의 validation 비교이므로 최종 test 결과나 통계적 유의성을 뜻하지 않는다.
재개 시 전체 RNG 상태는 복원하지 않아 중단 없는 실행과 bit-identical하지는 않다.
