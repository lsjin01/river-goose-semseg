# Labeling_Data_v2 7-class model bundle

이 폴더는 완료된 첫 번째 Labeling_Data_v2 7-class 실험의 최고 체크포인트 세 개를
구분 가능한 이름으로 모은 로컬 전달용 묶음이다.

- `teacher_dinov3_vitl16_mask2former`: 외부 사전학습에서 시작해 Labeling_Data_v2로
  학습한 Teacher
- `student_segformer_b1_kd`: 위 Teacher로 지식 증류한 B1
- `student_segformer_b0_kd`: 위 Teacher로 지식 증류한 B0

`*.pt`는 원본 체크포인트의 하드링크이므로 추가 디스크 공간을 거의 사용하지 않는다.
폴더를 다른 디스크이나 서버로 복사하면 정상적인 독립 파일로 복사된다. 모델 파일은
용량 때문에 `.gitignore` 대상이며 GitHub에는 README와 `manifest.json`만 올라간다.

`manifest.json`에 파일 크기, SHA-256, 선택 epoch, validation/test 성능과 원본 경로가
기록된다. 현재 진행 중인 multi-river 초기화 Teacher 파인튜닝 결과는 완료 후 별도의
최종 Teacher/KD 묶음으로 갱신해야 하며, 이번 세 파일과 혼동하면 안 된다.
