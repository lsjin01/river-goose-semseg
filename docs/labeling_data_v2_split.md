# Labeling_Data_v2 train/validation/test 분할

이 문서는 확장 데이터 `Labeling_Data_v2`를 학습·검증·테스트로 나눈 실제 기준을
기록한다. 원본 영상과 COCO annotation은 Git에 포함하지 않으며, 분할 생성 코드는
[`tools/prepare_labeling_binary_v2.py`](../tools/prepare_labeling_binary_v2.py)에 있다.

## 사용 데이터와 최종 개수

- 원본: `Labeling_Data (1).zip`을 해제한 `Labeling_Data_v2`
- 원본 프레임: 3,986장, annotation task 58개
- 제외: `Altum/Images/task85/IMG_0107.tif` 1장
  - 유효 polygon label이 없으므로 음성 샘플로 가정하지 않고 제외했다.
- 최종 분할 대상: 3,985장
- seed: 42

| Split | 전체 | Altum | P1 | Task | 관리 그룹 |
|---|---:|---:|---:|---:|---:|
| Train | 2,790 | 1,154 | 1,636 | 45 | 23 |
| Validation | 572 | 247 | 325 | 7 | 6 |
| Test | 623 | 237 | 386 | 6 | 5 |
| **합계** | **3,985** | **1,638** | **2,347** | **58** | **34** |

`task`는 라벨링 작업 단위이며 반드시 실제 촬영 episode 또는 서로 독립된 장소를
뜻하지 않는다. `관리 그룹`은 아래 누수 방지 규칙으로 연결된 task들의 집합이다.

## 분할 절차

1. Altum과 P1의 COCO JSON을 읽고 category 숫자 ID가 아니라 **category 이름**으로
   라벨을 해석한다. 일부 task는 category ID 체계가 서로 다르기 때문이다.
2. 유효한 polygon label이 없는 프레임은 제외한다. 미라벨을 `non-algae`로 취급하지
   않는다.
3. 각 원본 영상의 SHA256을 계산한다. 내용이 같은 영상이 여러 task에 있으면 해당
   task들을 같은 관리 그룹으로 연결한다. 중복 영상을 제거할 때는 최종 annotation
   geometry까지 동일한지 확인한다.
4. GPS가 있는 P1 task끼리는 프레임 위치의 최소 거리가 **1 km 미만**이면 같은 관리
   그룹으로 연결한다. 날짜나 task 번호가 달라도 가까운 위치면 split을 나누지 않는다.
5. Altum에는 사용할 수 있는 GPS가 없으므로 task 단위 독립성만 적용한다. 따라서
   Altum task 사이 또는 Altum-P1 사이의 동일 장소 여부는 보장할 수 없다.
6. 관리 그룹을 깨지 않은 상태에서 seed 42로 120,000개의 70/15/15 후보를 만들고,
   세 split 모두에 Altum/P1과 binary 양성/음성이 존재하는 후보만 허용한다.
   프레임·센서 비율과 클래스 존재 비율이 목표 비율에 가장 가까운 후보를 고정한다.
7. 입력 정규화 통계는 **train 영상만** 사용해 센서별로 계산한다.
8. 생성 후 source path, SHA256, task, 관리 그룹이 split 사이에 겹치지 않는지 다시
   검증한다.

검증 결과는 다음과 같다.

| 검사 | 결과 |
|---|---:|
| Task overlap | 0 |
| Exact SHA256 overlap | 0 |
| GPS 관리 그룹 overlap | 0 |
| 정규화 통계 산출 split | train only |

## 고정된 관리 그룹 배정

여러 task가 쉼표로 연결된 행은 동일 영상 또는 P1 GPS 1 km 규칙으로 한 관리
그룹이 된 경우다. 프레임 수는 해당 관리 그룹 전체의 합이다.

### Train

| Sensor | 관리 그룹에 포함된 task | 프레임 |
|---|---|---:|
| Altum | 18 | 78 |
| Altum | 22 | 41 |
| Altum | 24 | 16 |
| Altum | 27 | 122 |
| Altum | 32 | 25 |
| Altum | 33 | 54 |
| Altum | 35 | 118 |
| Altum | 42 | 108 |
| Altum | 45 | 101 |
| Altum | 46 | 132 |
| Altum | 47 | 28 |
| Altum | 91 | 23 |
| Altum | 92 | 25 |
| Altum | 94 | 112 |
| Altum | 97 | 108 |
| Altum | 98 | 63 |
| P1 | 10, 23, 29, 30, 34 | 241 |
| P1 | 11, 28, 40, 41 | 540 |
| P1 | 7, 8, 21, 50, 51, 52 | 213 |
| P1 | 9, 26, 31, 48, 49, 89, 90 | 295 |
| P1 | 37, 38 | 163 |
| P1 | 53, 55 | 78 |
| P1 | 56, 57, 61 | 106 |

### Validation

| Sensor | 관리 그룹에 포함된 task | 프레임 |
|---|---|---:|
| Altum | 25 | 69 |
| Altum | 44 | 117 |
| Altum | 85 | 61 |
| P1 | 20, 60 | 114 |
| P1 | 76 | 2 |
| P1 | 86 | 209 |

### Test

| Sensor | 관리 그룹에 포함된 task | 프레임 |
|---|---|---:|
| Altum | 36 | 30 |
| Altum | 43 | 17 |
| Altum | 95 | 190 |
| P1 | 39 | 212 |
| P1 | 62, 63 | 174 |

## 재현 명령

출력 폴더는 비어 있거나 존재하지 않아야 한다.

```bash
python tools/prepare_labeling_binary_v2.py \
  --source Labeling_Data_v2 \
  --output data/labeling_binary_v2_geo
```

이 실행은 `manifest.json`과 `verification.json`을 생성한다. 현재 고정 manifest의
SHA256은 다음과 같다.

```text
c789b6a7356573bf6b3035f71e3862a409f14690823738bd636087b0396229b5
```

7-class 실험은 영상을 다시 분할하지 않고 위 split을 정확히 재사용해 라벨만
`river/land/bridge/other/nps/turbid/algae_including_nps_algae`로 재매핑한다.

```bash
python tools/prepare_merged_algae_7class.py \
  --source data/labeling_binary_v2_geo \
  --output data/labeling_merged_algae_7class_v2_geo
```

7-class manifest SHA256:

```text
50c7e6d23a33d31159e200d136f204aa82f4cb6bee02710499f9f21f1c85b424
```

두 manifest에는 원본 경로와 annotation 전체가 들어가므로 Git에는 올리지 않는다.
대신 생성 코드, 고정 seed, 실제 그룹 배정, 예상 개수와 manifest hash를 이 문서에
기록한다.

## 해석상의 한계

- P1은 GPS가 있어 확인 가능한 1 km 인접 촬영을 묶었지만, 1 km보다 멀어도 같은
  하천 환경일 수 있다.
- Altum에는 GPS가 없어 task 단위 분리만 검증할 수 있다.
- 센서가 다른 Altum과 P1의 동일 장소 여부는 확인하지 못했다.
- 그룹별 프레임 수 차이가 커서 정확히 70/15/15가 아니라 70.01/14.35/15.63%다.
- `river`는 7-class 기준 train 한 프레임에만 있고 validation/test에는 없어 해당
  클래스의 일반화 성능을 측정할 수 없다.
