# 랩미팅 발표 가이드: 확장 하천 영상 데이터의 Binary Algae Segmentation

> 작성일: 2026-09-15 (KST)<br>
> 실험 상태: 학습 진행 중. 이 문서의 정량 결과는 **epoch 30까지의 validation 중간 결과**다.<br>
> 핵심 문장: **확장 데이터 3,985개 라벨 프레임을 장면 그룹 단위로 나누고, algae 계열과 nps_algae를 하나의 양성 클래스로 정의하여 DINOv3 ViT-L + Mask2Former를 학습하고 있다. 현재 validation 최고 2-class mIoU는 94.68%지만, 일부 P1 task의 성능이 매우 낮아 일반화 성능을 확정하기 전 하위 그룹 분석이 필요하다.**

---

## 0. 발표 전에 먼저 구분해야 할 세 가지

교수님께 설명할 때 다음 세 가지를 섞지 않는 것이 가장 중요하다.

1. **과제 정의:** 무엇을 algae라고 부르는가?
2. **데이터 분할의 독립성:** train/validation/test가 정말 다른 장면인가?
3. **모델 성능:** 어느 split, 어느 metric, 어느 checkpoint의 숫자인가?

현재 발표할 수 있는 결론은 다음과 같다.

- 현재 문제는 녹조의 세부 단계 분류가 아니라 **픽셀 단위 이진 영역 분할**이다.
- 현재 94.68%는 **validation mIoU**이며 최종 test 점수가 아니다.
- P1은 GPS 기반으로 가까운 task를 묶었지만, Altum은 GPS가 없어 task 단위 분리까지만 보장한다.
- `algae0`와 `nps_algae`를 양성에 넣는 것은 이번 실험의 **운영적 라벨 정의**다. 생물학적 정의로 검증된 것은 아니다.

---

## 1. 발표 권장 순서

아래 순서로 설명하면 질문이 들어와도 논리 흐름이 무너지지 않는다.

1. 기존 11-class 문제에서 무엇이 어려웠는가?
2. 이번 binary 연구 질문은 무엇인가?
3. 새 데이터는 어디서 왔고 기존 데이터와 어떤 관계인가?
4. 양성·음성 라벨은 어떻게 정의했는가?
5. 프레임이 아닌 장면 단위로 어떻게 분할했는가?
6. RGB와 Altum 스펙트럼 페이지는 어떻게 입력되는가?
7. DINOv3와 Mask2Former는 각각 무슨 일을 하는가?
8. 학습 설정과 체크포인트 선택 기준은 무엇인가?
9. 현재 validation 결과는 어떠한가?
10. 어떤 장면에서 실패하며, 무엇을 아직 주장할 수 없는가?
11. 다음 실험으로 무엇을 검증할 것인가?

---

## 2. 30초 요약

다음 문장을 그대로 발표 도입에 사용할 수 있다.

> 기존 데이터에서는 algae0부터 algae4까지 세부 단계를 분리했지만, 단계 간 시각적 혼동이 매우 컸습니다. 그래서 이번에는 algae 계열과 nps_algae를 하나의 관심 영역으로 묶어, algae 관련 영역과 나머지를 구분하는 binary semantic segmentation으로 문제를 재정의했습니다. 새 데이터는 기존 795프레임을 모두 포함하는 확장본이며, 라벨이 있는 3,985프레임을 사용합니다. 프레임 무작위 분할은 하지 않았고, 같은 task와 P1의 GPS 1km 이내 task를 같은 그룹으로 묶었습니다. 현재 DINOv3 ViT-L과 Mask2Former를 학습 중이며, epoch 30 validation mIoU가 94.68%입니다. 다만 task별 편차가 커서 최종 test 평가 전에 실패 장면과 분할 한계를 추가로 점검해야 합니다.

---

## 3. 연구 질문과 문제 재정의

### 3.1 기존 문제

기존 11-class 설정의 클래스는 다음과 같았다.

```text
land, bridge, other, nps,
algae0, algae1, algae2, algae3, algae4,
turbid, nps_algae
```

기존 모델은 algae 영역 자체는 비교적 잘 찾았지만 단계 구분에서 다음과 같은 혼동을 보였다.

- `algae1 → algae2`: 81.85%
- `algae3 → algae2`: 68.00%
- `algae4 → nps`: 62.72%

이 비율은 이전 795프레임 데이터의 test에서 각 정답 클래스 픽셀 중 해당 클래스로 잘못 예측한 비율이다.
새 확장 데이터의 현재 validation 결과와 직접 비교하는 숫자가 아니다.

### 3.2 이번 연구 질문

이번 질문은 다음과 같다.

> **세부 녹조 단계가 아니라 algae 관련 영역 전체를 하나의 관심 대상으로 정의했을 때, 서로 다른 센서의 하천 영상에서 해당 영역을 픽셀 단위로 안정적으로 분할할 수 있는가?**

출력은 픽셀마다 두 클래스 중 하나다.

| ID | 학습 클래스 | 포함하는 원본 라벨 |
|---:|---|---|
| 0 | `non_algae` | river, land, bridge, other, nps, turbid |
| 1 | `algae_including_nps_algae` | algae, algae0, algae1, algae2, algae3, algae4, nps_algae |
| 255 | ignore | ambiguous 및 평가 제외 패딩 |

### 3.3 반드시 말해야 하는 라벨 의미의 한계

- `algae0`가 생물학적으로 녹조가 존재하는 상태라는 근거는 아직 라벨링 지침으로 확인하지 못했다.
- `nps_algae`는 구성 픽셀을 algae와 NPS로 나눌 근거가 없어 사용자의 연구 목적에 따라 전체를 양성으로 포함했다.
- `nps`의 구체적인 영상 판정 기준도 원본 라벨링 지침으로 확인하지 못했다.
- 따라서 현재 양성 클래스는 **연구 목적상 정의한 algae-related region**이지, 현장 수질 측정으로 검증된 녹조 농도 또는 생물학적 bloom 판정이 아니다.

교수님께서 “왜 algae0까지 녹조인가?”라고 물으면 다음과 같이 답한다.

> 이번 실험에서는 데이터 제공자가 구분한 algae 계열 전체와 nps_algae를 관심 영역으로 보는 운영적 정의를 사용했습니다. 다만 algae0의 실제 의미를 확인하지 못했기 때문에, 최종 논문 정의 전에 라벨링 가이드 확인과 algae0 제외 ablation이 필요합니다.

---

## 4. 데이터셋

### 4.1 원본과 확장본의 관계

- 작은 원본: `Labeling_Data.zip`, 795프레임, 17개 annotation task.
- 확장본: `Labeling_Data (1).zip`, 3,986프레임, 58개 annotation task.
- 공통 이미지와 annotation JSON을 SHA256으로 비교했다.
- 확장본은 기존 이미지 795개와 라벨 JSON 17개를 내용까지 동일하게 포함한다.
- 기존 파일을 추가로 합치면 중복되므로 확장본만 사용한다.
- 작은 실제 데이터와 ZIP은 정리했고, 이전 manifest 호환을 위해 `Labeling_Data` 경로는 확장 폴더를 가리키는 링크로 유지했다.

### 4.2 사용 프레임

| 센서 | 원본 프레임 | 사용 프레임 | 비고 |
|---|---:|---:|---|
| Altum | 1,639 | 1,638 | 라벨 없는 1장 제외 |
| P1 | 2,347 | 2,347 | 모두 사용 |
| **합계** | **3,986** | **3,985** | |

제외한 파일은 `Altum/Images/task85/IMG_0107.tif`이다. 이 파일에는 non-ignore polygon 라벨이 없다.
라벨이 없는 이미지를 `non_algae`로 간주하면 false negative가 포함된 잘못된 음성 데이터가 될 수 있으므로 제외했다.

### 4.3 라벨 스키마가 두 종류였던 문제

58개 JSON 중 55개는 12-category 스키마였고, P1의 3개 task는 14-category 스키마였다.
같은 숫자 ID가 서로 다른 의미를 가질 수 있으므로 **category ID가 아니라 category name을 기준으로** binary `label_id`를 생성했다.

- 전체 22,221개 annotation의 이름 기반 변환을 검사했다.
- polygon 좌표가 annotation 이미지 범위를 벗어난 사례는 0개였다.
- 원본 이미지와 COCO JSON은 수정하지 않았다.
- 변환된 `label_id`는 새 manifest에만 기록했다.

### 4.4 프레임과 task를 어떻게 이해하는가?

- **프레임:** 개별 TIFF 또는 JPEG 이미지 한 장.
- **task:** 하나의 annotation JSON이 담당하는 이미지 묶음.
- **관리 그룹:** 같은 장면일 가능성이 있는 여러 task를 누수 방지를 위해 합친 단위.
- task가 반드시 독립적인 비행 episode 또는 독립 장소라는 메타데이터는 없다.

### 4.5 교수님께 보여드릴 센서별 원본·라벨 예시

아래 예시는 모두 **train split에서만** 선택했다. 예약된 test 이미지는 열지 않았다.
색상 오버레이에는 binary 통합 전의 원본 다중 클래스 이름을 사용했다.

#### P1 RGB 예시

각 행은 왼쪽부터 `Image`, `Overlay`, `GT (Segmentation)` 순서다.
P1의 Image는 원본 RGB이고 Overlay는 동일 영상에 원본 다중 클래스 정답을 겹친 것이다.
페이지 아래의 범례에서 색상별 원본 클래스를 확인할 수 있다.
GT에서 검은색은 semantic class가 아니라 polygon이 없는 `unlabeled/ignore` 영역이다.
서로 다른 task에서 장면을 고르고, `algae0~4`, `nps_algae`, `turbid`, `bridge`, `river`처럼
흔하지 않은 라벨도 전체 패널에 포함되도록 구성했다.

로컬 생성 파일: `docs/assets/lab_meeting_sensor_samples/p1_diverse_samples.png`

#### Altum 예시

Altum 원본은 7페이지 uint16 TIFF이므로 화면에 그대로 표시할 수 없다.
각 행은 왼쪽부터 `Image`, `Overlay`, `GT (Segmentation)` 순서다.
Image는 각 페이지를 독립적으로 1–99 percentile stretch한 뒤
`page2/page1/page0 → R/G/B`로 배치한 **발표용 표시 합성**이다.
Overlay는 이 합성 영상과 정답을 겹친 것이며, GT는 클래스 색만 표시한다.
페이지 아래의 범례에서 각 색의 원본 클래스를 확인할 수 있다.
GT에서 검은색은 semantic class가 아니라 `unlabeled/ignore` 영역이다.
이 합성은 모델의 실제 정규화 방식이 아니며, page별 물리적 파장 순서가 검증됐다는 뜻도 아니다.

로컬 생성 파일: `docs/assets/lab_meeting_sensor_samples/altum_diverse_samples.png`

Altum 한 프레임의 TIFF 페이지 구성을 다음 그림으로 보여준다.
page0~5는 서로 다른 장면 신호를 보이지만, 표본으로 확인한 모든 Altum task에서 page6은
거의 전체가 값 `65534`인 NoData형 페이지였다. 모든 페이지는 보기 위한 목적으로 각각 대비를
늘렸으므로 페이지 간 절대 밝기를 비교하면 안 된다.

로컬 생성 파일: `docs/assets/lab_meeting_sensor_samples/altum_seven_page_overview.png`

원본 데이터에서 만든 이미지와 선택 목록은 공개 Git에 포함하지 않는다. 다음 명령으로
train split에서만 동일한 자료를 로컬 생성한다.

```bash
python tools/make_lab_sensor_samples.py
```

---

## 5. Train / Validation / Test 분할

### 5.1 최종 수량

| Split | 프레임 | Altum | P1 | Task | 관리 그룹 | 전체 평가 타일 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 2,790 | 1,154 | 1,636 | 45 | 23 | 매 epoch 무작위 crop 2,790개 |
| Validation | 572 | 247 | 325 | 7 | 6 | 27,074 |
| Test | 623 | 237 | 386 | 6 | 5 | 35,390 |

전체 관리 그룹은 34개다.

### 5.2 장면 누수 방지 규칙

프레임을 무작위로 70/15/15로 나누지 않았다.

1. 같은 annotation task의 모든 프레임을 하나로 유지했다.
2. 동일 이미지 SHA256이 발견되면 같은 그룹으로 연결하도록 했다.
3. P1은 2,347개 프레임 모두 GPS가 있었다.
4. 서로 다른 P1 task라도 임의의 촬영 지점 간 최단거리가 1km 미만이면 연결했다.
5. 연결 관계의 연결요소 전체를 한 split에 넣었다. A–B와 B–C가 연결되면 A–C도 같은 그룹이다.
6. 그룹 배치는 약 70/15/15의 프레임·센서·binary 클래스 이미지 수를 맞추도록 seed 42로 탐색했다.
7. 모델 예측이나 validation 성능은 분할 생성에 사용하지 않았다.

검증 결과:

- task overlap: 0
- exact image SHA256 overlap: 0
- 확인된 P1 GPS 관리 그룹 overlap: 0
- train/validation/test 모두 Altum·P1과 두 binary 클래스를 포함
- 입력 정규화 통계는 train에서만 계산

### 5.3 분할에서 아직 보장하지 못하는 것

- Altum 1,638개 프레임은 GPS와 신뢰할 촬영 시각이 없어 서로 다른 task가 같은 장소인지 확인할 수 없다.
- Altum과 P1 사이의 동일 장소 여부도 연결하지 못했다.
- 1km는 보수적인 관리 임계값이지, 하천의 시각적 상관 길이를 과학적으로 최적화한 값은 아니다.
- validation은 관리 그룹 6개, test는 5개뿐이다. 프레임 수가 많아도 독립 장면 수는 제한적이다.
- 따라서 pixel-level 신뢰구간을 프레임이 모두 독립이라는 가정으로 계산하면 안 된다.

권장 표현:

> P1은 GPS 1km 연결 그룹 기반 분할이고, Altum은 task 기반 분할이다. 프레임 무작위 분할은 아니지만 모든 센서에 대해 물리적 장소 독립성을 완전히 증명한 것은 아니다.

---

## 6. 멀티모달·스펙트럼 입력

### 6.1 현재 입력의 의미

Altum과 P1을 같은 장면 쌍으로 맞춰 동시에 넣는 paired multimodal 구조가 아니다.
각 센서 이미지는 독립적인 샘플이며, 하나의 모델 파라미터를 공유한다.

```text
Altum 한 장: TIFF 7페이지 ───────┐
                                ├→ 같은 9-slot 데이터 인터페이스
P1 한 장: RGB 3채널 + 0 padding ┘
```

### 6.2 로더 출력 9개 슬롯

모델에 전달되는 텐서 크기는 `[B, 9, 768, 768]`이다.

| 슬롯 | Altum | P1 |
|---:|---|---|
| 0–2 | 정규화한 TIFF page0–2 | 정규화한 B, G, R |
| 3–5 | 정규화한 TIFF page3–5 | 0 |
| 6 | NoData형 page6도 현재 텐서에는 들어감 | 0 |
| 7 | `(page3-page2)/(page3+page2+ε)` | 0 |
| 8 | `(page3-page4)/(page3+page4+ε)` | 0 |

현재 `fusion_type: input`에서는 어댑터가 마지막 비율 슬롯 2개를 제외하여 **실제로 page/RGB 슬롯 0–6만 사용**한다.
따라서 현재 모델이 9개 정보를 모두 활용한다고 말하면 안 된다.

추가 점검 결과, 22개 Altum task에서 뽑은 원본의 page6은 거의 전 픽셀이 `65534`였다.
즉 현재 구현은 7개 슬롯을 어댑터에 전달하지만 page6은 유효한 장면 밴드라기보다 NoData 상수에 가깝다.
실질적인 장면 신호는 page0~5에서 들어오며, 모델은 거의 상수인 page6을 학습 중 무시할 수 있다.

### 6.3 각 Altum page의 의미

프로젝트의 기존 변환 코드와 입력 어댑터는 Altum page0~5를 다음 순서로 정의한다.
이 순서는 MicaSense Altum 공식 imager 순서와도 일치한다. 다만 현재 TIFF 파일 자체에는
페이지 이름·중심 파장 태그가 남아 있지 않으므로, 아래 표는 **프로젝트가 사용한 working mapping**이다.
최종 논문에는 원본 TIFF 생성 명세 또는 센서 원본 파일로 한 번 더 확인해야 한다.
공식 사양은 [MicaSense Altum Integration Guide](https://support.micasense.com/hc/en-us/articles/360010025413-Altum-Integration-Guide)와
[MicaSense 센서 비교표](https://support.micasense.com/hc/en-us/articles/1500007828482-Comparison-of-MicaSense-Cameras)를 참고한다.

| TIFF page | 프로젝트상 밴드 | 대표 중심 파장 | 무엇을 측정하는가? | 녹조 분석에서 기대하는 정보 |
|---:|---|---:|---|---|
| 0 | Blue | 약 475 nm | 청색 가시광 반사 | 물·대기 산란, 탁도 및 부유물 변화에 민감하지만 수중 감쇠도 큼 |
| 1 | Green | 약 560 nm | 녹색 가시광 반사 | 녹색으로 보이는 조류·식생과 물의 밝기 차이 표현 |
| 2 | Red | 약 668 nm | 적색 가시광 반사 | 엽록소 흡수와 관련된 대비; NIR과 함께 식생형 물질 구분에 사용 가능 |
| 3 | NIR | 약 840–842 nm | 근적외선 반사 | 깨끗한 물은 강하게 흡수하고 수면 위 식생·두꺼운 부유 녹조는 상대적으로 밝을 수 있음 |
| 4 | Red Edge | 약 717 nm | Red와 NIR 사이의 급격한 반사 변화 | 엽록소·생체량 변화에 민감할 가능성이 있어 algae 단계 구분 후보 정보 |
| 5 | Thermal/LWIR | 약 11 μm | 가시광 반사가 아닌 표면 열복사 | 수온·표면 온도 차이를 통한 간접 정보; 녹조 자체를 직접 측정하는 채널은 아님 |
| 6 | NoData형 page | 해당 없음 | 거의 전 픽셀 `65534` | 장면 정보가 없으므로 분석 대상 밴드로 보면 안 됨 |

여기서 `Blue + Green + Red` 세 밴드를 다음 순서로 화면에 배치하면 사람이 보는 RGB와 비슷한
표시 영상을 만들 수 있다.

```text
표시 영상 R <- page2 (Red)
표시 영상 G <- page1 (Green)
표시 영상 B <- page0 (Blue)
```

하지만 각 밴드를 별도로 percentile stretch했기 때문에 이것은 방사보정된 자연색 사진이 아니라
**보기 쉽게 만든 색 합성**이다.

현재 로더가 계산하는 두 비율은 위 working mapping이 맞다는 가정에서 다음처럼 해석된다.

```text
(page3 - page2) / (page3 + page2 + eps)
= (NIR - Red) / (NIR + Red + eps)       # NDVI와 같은 형태

(page3 - page4) / (page3 + page4 + eps)
= (NIR - RedEdge) / (NIR + RedEdge + eps)
```

단, 현재 `fusion_type: input` 실험은 이 두 비율 슬롯을 입력 어댑터 직전에 제거하므로
**현재 모델은 계산된 지수를 직접 사용하지 않고 page0~6만 받는다.** 또한 육상 식생과 달리
수중 조류는 물의 NIR 흡수, 수면 반사와 햇빛 반짝임 영향을 크게 받으므로 NDVI형 값이 항상
녹조 농도와 비례한다고 가정하면 안 된다.

#### NIR·Red Edge·Thermal 실제 샘플

아래 그림들은 각 밴드마다 동일한 Altum train 대표 장면 2개만 사용한다. 각 행은 왼쪽부터
`RGB 유사 합성`, `해당 밴드 grayscale`, `해당 밴드 false color + 원본 GT`다.
각 프레임을 1–99 percentile로 독립 변환했으므로 장면 사이 색을 절대 반사율·온도로 비교하면 안 된다.

**NIR — page3 working mapping**

로컬 생성 파일: `docs/assets/lab_meeting_sensor_samples/altum_nir_samples.png`

**Red Edge — page4 working mapping**

로컬 생성 파일: `docs/assets/lab_meeting_sensor_samples/altum_red_edge_samples.png`

**Thermal/LWIR — page5 working mapping**

로컬 생성 파일: `docs/assets/lab_meeting_sensor_samples/altum_thermal_lwir_samples.png`

그림의 GT 색은 밴드값과 클래스 위치를 함께 보기 위한 반투명 표시다.
Thermal 색은 온도처럼 보이지만 보정식과 단위를 적용하지 않은 원시값의 상대적 크기이며 섭씨가 아니다.

### 6.4 정규화

센서별·채널별로 train에서만 1st/99th percentile과 표준편차를 계산한다.

```text
scale = max(p99 - p01, standard deviation, 1)
normalized = asinh((value - p01) / scale)
```

- hard clipping 대신 `asinh`로 큰 값을 완만하게 압축한다.
- P1의 존재하지 않는 추가 채널은 정규화 이후 다시 0으로 고정한다.
- P1 원본이 annotation grid보다 두 배 큰 경우 좌표 비율에 맞춰 source crop을 resize한다.

### 6.5 입력 어댑터

현재는 7채널을 DINOv3가 받을 수 있는 3채널 표현으로 학습해 변환한다.

```text
7채널 ─┬─ Conv 1×1: 7 → 3 ──────────────────────┐
       └─ Conv 1×1: 7 → 16 → GELU → 3 ─────────┤ 더하기
                                                 ↓
                                       ImageNet 정규화
                                                 ↓
                                           DINOv3
```

- 초기에는 슬롯 2/1/0을 선택해 3채널처럼 시작한다.
- 추가 채널 가중치는 처음 0이지만 학습 중 갱신될 수 있다.
- 1×1 Conv이므로 이 어댑터는 같은 픽셀 위치의 채널만 섞는다.
- 7→3 압축이므로 정보 병목이 생길 수 있다.
- 유용한 채널을 실제로 사용했는지는 채널 제거 ablation이나 학습 가중치·민감도 분석 없이 확정할 수 없다.

### 6.6 스펙트럼 해석의 한계

프로젝트 코드는 page0~5를 Blue, Green, Red, NIR, Red Edge, Thermal로 정의하며 공식 센서 순서와
일치하지만, 현재 TIFF 자체에서 이를 증명하는 메타데이터는 확인하지 못했다.
따라서 이 매핑을 원본 생성 명세로 검증하기 전에는 물리적으로 확정된 측정값이라고 과장하면 안 된다.
현재 데이터는 “7페이지 TIFF 컨테이너이며 page0~5에는 장면 신호가 있고 page6은 NoData형”이라고
표현하는 것이 가장 정확하다.

다섯 가지 fusion 방법의 상세 비교는 [스펙트럼 결합 방법 README](spectral_fusion/README.md)를 참고한다.

---

## 7. 모델 구조

### 7.1 전체 흐름

```text
9-slot loader output
        ↓  (input 방식은 비율 2개 제외)
7→3 SpectralInputAdapter
        ↓
DINOv3 ViT-L/16 backbone
        ↓  24개 block 중 4, 11, 17, 23의 feature
공간 feature adapter / channel aligner
        ↓
Mask2Former pixel decoder + transformer decoder
        ↓
픽셀별 non_algae / algae_related semantic prediction

DINO CLS token ─→ 2-output CLS auxiliary head
```

### 7.2 구성 요소의 역할

- **DINOv3 ViT-L/16:** 입력을 16×16 patch로 보고 시각 특징을 추출하는 백본.
- **Spatial adapter:** 여러 해상도의 feature map을 구성해 segmentation decoder가 사용할 수 있도록 한다.
- **Mask2Former:** query별 class와 mask를 예측하고 semantic score로 합쳐 픽셀 라벨을 만든다.
- **CLS auxiliary head:** crop 전체에 각 binary 클래스가 존재하는지 보조적으로 예측한다.

### 7.3 파라미터 수

실제 optimizer 그룹 기준 학습 파라미터는 약 **340.35M**이다.

| 그룹 | 파라미터 수 | 학습률 |
|---|---:|---:|
| DINOv3 backbone | 303,129,600 | 1e-6 |
| 입력 어댑터·공간 어댑터·Mask2Former·CLS head 등 | 37,217,357 | 4e-5 |
| **합계** | **340,346,957** | |

두 GPU를 사용한다고 파라미터가 반씩 분할되는 것은 아니다.
현재 `torch.nn.DataParallel`은 모델을 두 GPU에 복제하고 batch를 나눠 처리한다.

### 7.4 사전학습 초기화

- DINOv3: `facebook/dinov3-vitl16-pretrain-lvd1689m`
- Mask2Former: `facebook/mask2former-swin-large-ade-semantic`
- 기존 11-class fine-tuned checkpoint에서 이어 학습하지 않았다.
- binary class predictor는 출력 크기가 달라 새로 초기화했다.
- 모든 학습 가능한 DINO 파라미터에 gradient가 흐르는지 두 GPU smoke test에서 확인했다.

이전 DataParallel 코드에서는 백본 gradient가 차단되는 문제가 있었지만 현재 코드는 수정됐고,
이번 실행 전 smoke test에서 DINO gradient norm `2.0586`이 0이 아님을 확인했다.

---

## 8. 학습 방식

### 8.1 핵심 하이퍼파라미터

| 항목 | 값 |
|---|---:|
| 최대 epoch | 100 |
| GPU | 0, 1 |
| 타일 크기 | 768×768 |
| 총 micro-batch | 4 |
| gradient accumulation | 3 |
| 명목상 effective batch | 12 |
| 일반 학습률 | 4e-5 |
| DINOv3 학습률 | 1e-6 |
| weight decay | 0.02 |
| scheduler | PolynomialLR, power 0.9 |
| warmup | optimizer update 200회 |
| 좌우 반전 확률 | 0.5 |
| class-aware crop | 사용 안 함 (`prob=0`) |
| random seed | 42 |
| validation 간격 | 5 epoch |
| early stopping | validation 4회 연속 개선 없음 |
| checkpoint 선택 | validation 2-class mIoU |
| TTA | 학습/validation 모두 없음 |

### 8.2 한 epoch의 의미

Train dataset의 `__len__`은 프레임 수 2,790이다.
각 epoch마다 각 프레임에서 768×768 random crop 하나를 가져온다.

- `drop_last=True`이고 batch 4이므로 epoch당 697 micro-batch, 2,788 crop을 처리한다.
- gradient accumulation 3이므로 epoch당 약 233 optimizer update다.
- 큰 8192×5460 P1 프레임 전체를 한 epoch에 모두 보는 방식이 아니다.
- validation은 572프레임 전체를 27,074개의 겹치지 않는 768 타일로 평가한다.
- 가장자리 패딩은 ignore=255라 metric에 포함되지 않는다.

교수님께서 “프레임 수가 2,790인데 왜 batch가 697인가?”라고 물으면 `2790 / 4`, 마지막 불완전 batch 제외라고 답한다.

### 8.3 손실 함수

Mask2Former set criterion을 사용한다.

- query classification CE weight: 2.0
- mask BCE weight: 5.0
- Dice loss weight: 5.0
- 학습 point sampling: 12,544 points
- no-object weight: 0.1
- decoder auxiliary layer들의 CE/mask/Dice loss도 합산
- CLS presence BCE × 0.05 추가

로그의 총 loss가 7~15처럼 보여도 단일 픽셀 CE가 그 값이라는 뜻이 아니다.
여러 decoder layer의 가중 loss가 합산되므로, loss 절댓값보다 같은 설정 내 추세와 mIoU를 함께 본다.

### 8.4 CLS-aux의 정확한 의미

각 crop에 `non_algae` 픽셀과 `algae_related` 픽셀이 존재하는지를 2개 independent sigmoid 출력으로 예측한다.
두 클래스가 한 crop에 함께 있을 수 있으므로 softmax 단일 선택이 아니라 multi-label BCE다.
이 보조 분류 결과가 최종 픽셀 segmentation을 대신하는 것은 아니다.

---

## 9. 평가 지표

각 클래스 `c`의 IoU는 다음과 같다.

```text
IoU_c = TP_c / (TP_c + FP_c + FN_c)
mIoU  = (IoU_non_algae + IoU_algae_related) / 2
```

- 전체 validation confusion matrix를 먼저 누적한 뒤 두 클래스 IoU를 계산한다.
- 프레임별 IoU를 단순 평균한 값이 아니다.
- 픽셀이 많은 장면이 전체 confusion에 더 크게 기여한다.
- 그래서 전체 mIoU 외에 센서별·task별 결과를 반드시 함께 봐야 한다.
- pixel accuracy는 큰 영역 클래스에 지배될 수 있으므로 주 지표로 사용하지 않는다.

---

## 10. 현재 학습 결과

### 10.1 epoch별 결과

| Epoch | Train mIoU | Validation mIoU | Best validation | Val CLS-aux accuracy |
|---:|---:|---:|---:|---:|
| 5 | 95.94% | 92.92% | 92.92% | 94.53% |
| 10 | 96.84% | 93.61% | 93.61% | 95.15% |
| 15 | 96.99% | 91.85% | 93.61% | 95.07% |
| 20 | 97.45% | 92.26% | 93.61% | 95.42% |
| 25 | 97.45% | 92.98% | 93.61% | 95.38% |
| **30** | **97.62%** | **94.68%** | **94.68%** | **95.86%** |

현재 best checkpoint:

```text
outputs/labeling_binary_algae_v2/binary_aux005_seed42/
best_epoch_030_miou_0.9468.pt
```

이 결과는 아직 최종 결과가 아니다. 최대 100 epoch이고, 5 epoch마다 validation하며 4회 연속 개선이 없으면 조기 종료한다.

### 10.2 epoch 30 전체 validation 클래스별 결과

| 클래스 | IoU | Precision | Recall |
|---|---:|---:|---:|
| non_algae | 94.10% | 95.49% | 98.48% |
| algae + nps_algae | 95.27% | 98.79% | 96.39% |
| **mIoU** | **94.68%** | | |

전체 validation에서 algae 관련 false negative가 false positive보다 많다.

- 실제 non_algae를 algae로 예측: 91,042,384픽셀
- 실제 algae를 non_algae로 예측: 278,968,948픽셀

즉 현재 모델은 validation에서 algae를 과잉 검출하기보다 **일부 algae 영역을 놓치는 경향**이 더 크다.

### 10.3 센서별 결과

| 센서 | mIoU | non_algae IoU | algae IoU |
|---|---:|---:|---:|
| Altum | **98.00%** | 97.70% | 98.29% |
| P1 | **94.49%** | 93.90% | 95.09% |

Altum이 P1보다 약 3.5 percentage point 높다.
이 차이가 추가 채널의 효과라고 바로 결론 내릴 수는 없다. 장면 난이도, 촬영 조건, 해상도, 라벨 품질, split 구성이 동시에 다르기 때문이다.

### 10.4 task별 결과에서 드러난 핵심 문제

| Validation task | 프레임 | mIoU |
|---|---:|---:|
| Altum/task25 | 69 | 96.42% |
| Altum/task44 | 117 | 99.04% |
| Altum/task85 | 61 | 97.59% |
| P1/task20 | 포함 그룹 일부 | 98.10% |
| P1/task60 | 포함 그룹 일부 | 95.67% |
| **P1/task76** | **2** | **25.91%** |
| P1/task86 | 209 | 94.91% |

`P1/task76`의 세부 성능:

- non_algae IoU: 36.19%
- algae IoU: 15.62%
- algae precision: 78.78%
- algae recall: 16.31%

따라서 전체 94.68%만 보고 모든 장면에서 안정적이라고 말할 수 없다.
task76은 2프레임이라 전체 픽셀 누적 metric에는 영향이 제한적이지만, 새로운 촬영 조건에서의 failure case일 가능성이 있다.

발표 시 권장 표현:

> Aggregate validation 성능은 높지만 task별 최저 성능은 매우 낮습니다. 특히 task76은 표본이 두 장뿐이라 통계적 결론을 내리기 어렵지만, domain shift 또는 라벨/전처리 문제를 찾기 위한 우선 분석 대상으로 보고 있습니다.

---

## 11. 기존 92.46%와 현재 94.68%를 비교하면 안 되는 이유

이전에 보고한 92.46%는 다음 조건이었다.

- 기존 795프레임 데이터에서 학습한 **11-class 모델** 사용.
- 기존 test 114프레임의 11-class argmax를 사후 binary remap.
- 새로운 binary 모델을 학습한 결과가 아님.
- 과거 split의 test 결과.

현재 94.68%는 다음 조건이다.

- 확장 데이터 3,985프레임.
- 새로 만든 장면 그룹 split.
- 처음부터 2-class 출력으로 학습 중인 모델.
- 현재 split의 validation 결과.

데이터, split, 모델 출력, 평가 split이 모두 다르므로 `94.68 - 92.46 = 2.22%p 개선`이라고 주장하면 안 된다.

---

## 12. 현재 결과에서 합리적으로 말할 수 있는 것

### 말할 수 있는 것

- 이름 기반 binary 라벨 변환과 train-only normalization을 적용했다.
- 프레임 무작위가 아니라 task/P1-GPS 관리 그룹 단위로 분할했다.
- 현재 설정에서 epoch 30 validation 2-class mIoU는 94.68%다.
- 전체 validation에서는 두 클래스 IoU가 모두 94% 이상이다.
- Altum validation이 P1 validation보다 높다.
- P1 task76은 성능이 매우 낮아 subgroup robustness가 해결되지 않았다.

### 아직 말하면 안 되는 것

- 최종 test 성능이 94.68%다.
- 스펙트럼 때문에 RGB보다 정확해졌다.
- 모든 7페이지가 모델 성능에 기여한다.
- 모든 validation/test 장면이 물리적으로 완전히 독립이다.
- algae0와 nps_algae가 생물학적으로 동일한 녹조 클래스다.
- 다른 하천·계절·고도·센서에서도 같은 성능이 나온다.
- 92.46% 대비 2.22%p 개선했다.

---

## 13. 예상 질문과 답변

### Q1. 왜 binary로 바꿨는가?

세부 algae 단계 사이에 강한 혼동이 있었고, 우선 연구 목적이 단계 추정이 아니라 algae 관련 영역 탐지인지 확인하기 위해 문제를 단순화했다.
binary 성능이 높다고 세부 단계 분류 문제가 해결된 것은 아니다.

### Q2. 왜 nps_algae를 모두 양성으로 넣었는가?

현재 annotation은 복합 영역 내부를 NPS와 algae 픽셀로 분해하지 않는다.
이번 운영 목적에 따라 전체 복합 영역을 양성으로 포함했다. 이 결정의 영향은 nps_algae 제외 ablation으로 확인할 수 있다.

### Q3. algae0도 정말 녹조인가?

아직 라벨 가이드로 의미를 검증하지 못했다. 현재는 이름상 algae 계열을 통합한 실험 정의다.
최종 결론 전 algae0 의미 확인과 제외 실험이 필요하다.

### Q4. 데이터가 3,986장인데 왜 3,985장만 쓰는가?

Altum task85의 한 장에 polygon 라벨이 없다. 이를 음성으로 오해하지 않도록 제외했다.

### Q5. 기존 795장도 새 데이터에 중복으로 들어간 것 아닌가?

확장본이 기존 파일을 포함하지만 하나의 경로에 한 번씩 존재한다. 기존 데이터를 다시 합치지 않았다.
공통 이미지와 JSON은 SHA256이 동일함을 확인했다.

### Q6. 장면별로 나눈 것이 맞는가?

P1은 task와 GPS 1km 연결 그룹 단위다. Altum은 GPS가 없어 task 단위까지만 보장한다.
따라서 모든 센서의 물리적 장소 분리라고 과장하지 않는다.

### Q7. 왜 GPS 임계값이 1km인가?

가까운 비행을 서로 다른 split에 넣지 않기 위한 보수적인 누수 방지 기준이다.
과학적으로 최적화한 상관 거리라는 뜻은 아니며, 거리 임계값 민감도 분석은 아직 하지 않았다.

### Q8. 572장 validation인데 왜 27,074타일인가?

고해상도 원본 전체를 768×768의 겹치지 않는 타일로 덮는다. 한 프레임이 여러 타일을 생성한다.
가장자리 패딩은 ignore 처리한다.

### Q9. Train도 전체 영상을 모두 보는가?

아니다. 매 epoch 각 train 프레임에서 random crop 하나를 뽑는다.
따라서 큰 이미지의 모든 위치가 매 epoch 노출되는 것은 아니다.

### Q10. Altum TIFF의 유효 페이지와 파장은 무엇인가?

TIFF에는 7페이지가 있지만 현재 확인상 page0~5에 장면 신호가 있고 page6은 거의 모두 `65534`인
NoData형 페이지다. page0~5의 물리적 파장은 메타데이터로 검증하지 못했으므로,
센서 원본 명세와 TIFF 생성 전처리 기록 확인을 후속 작업으로 둔다.

### Q11. P1에는 없는 스펙트럼을 어떻게 처리하는가?

RGB 3개 슬롯만 채우고 추가 슬롯은 0으로 유지한다. 없는 정보를 추정하거나 생성하지 않는다.

### Q12. 이것이 진정한 multimodal fusion인가?

서로 정렬된 RGB와 스펙트럼 이미지 쌍을 동시에 융합하는 방식은 아니다.
센서별 독립 샘플을 공통 9-slot 인터페이스로 처리하는 heterogeneous sensor training에 가깝다.

### Q13. 왜 7채널을 3채널로 줄이는가?

RGB 사전학습 DINOv3를 활용하기 위해 학습 가능한 입력 어댑터를 둔다.
다만 정보 병목 가능성이 있어 late/gated/cross-attention 방식과 동일 조건 비교가 필요하다.

### Q14. 스펙트럼이 실제로 도움 된다는 증거가 있는가?

현재 binary 확장 데이터에서 RGB-only 대조군이 없으므로 아직 증명하지 못했다.
Altum 점수가 높다는 사실만으로는 장면 난이도 차이를 배제할 수 없다.

### Q15. 왜 train mIoU가 validation보다 높은가?

학습 데이터 적합과 domain/group 차이로 일반적인 gap이 있을 수 있다.
또 train은 random crop 누적, validation은 전체 영상 타일 누적이라 평가 표본 구성도 다르다.

### Q16. validation 값이 91.85%에서 94.68%로 흔들리는 이유는?

모델 업데이트에 따라 경계와 어려운 P1 장면 예측이 달라질 수 있고, validation은 일부 관리 그룹에 집중돼 있다.
최고 checkpoint를 validation으로 선택하되, 여러 seed와 group-wise 결과 없이 변동을 과도하게 해석하지 않는다.

### Q17. task76이 25.91%인데 전체가 94.68%인 이유는?

전체 metric은 모든 validation 픽셀의 confusion matrix를 합친 뒤 계산한다.
task76은 두 프레임뿐이라 큰 task보다 가중치가 작다. 따라서 macro group 평균도 별도로 보고해야 한다.

### Q18. loss가 12인데 mIoU가 94%인 것이 이상하지 않은가?

총 loss는 Mask2Former decoder 여러 층의 class/mask/Dice 가중 loss 합이다.
픽셀 error rate나 `1 - IoU`와 같은 스케일이 아니다.

### Q19. test는 언제 평가하는가?

학습과 validation checkpoint 선택이 끝난 뒤 한 번 평가한다.
다만 이전 실험에서 확장본에 포함된 기존 일부 장면의 결과를 이미 보았기 때문에 완전한 blind external test라고 주장하지 않는다.

### Q20. 다음으로 가장 필요한 실험은 무엇인가?

같은 split에서 RGB-only와 7-page input을 비교하고, task76 실패 원인을 확인하는 것이다.
그 뒤 nps_algae/algae0 정의 ablation과 여러 seed 평가를 진행한다.

---

## 14. 다음 실험 우선순위

### 필수 1 — 실패 사례 점검

- P1/task76 두 프레임의 원본·정답·예측 overlay 확인.
- annotation 좌표와 source crop resize 정합 확인.
- 촬영 고도, 색감, 흐림, 반사, 수면 상태가 train과 다른지 확인.
- label definition이 다른 task와 일관되는지 검토.

### 필수 2 — RGB-only 대조군

동일 split·seed·학습 schedule에서 모든 센서에 공통으로 존재하는 RGB-position 3채널만 사용한다.
현재 A 방식과 비교해야 추가 Altum 페이지의 기여를 주장할 수 있다.

### 필수 3 — 라벨 정의 ablation

다음 세 조건을 비교한다.

1. algae0~4 + nps_algae
2. algae1~4 + nps_algae (`algae0` 제외)
3. algae0~4 (`nps_algae` 제외 또는 ignore)

단, test를 반복해 정의를 선택하지 말고 validation 또는 사전 정의된 연구 목적에 따라 선택한다.

### 권장 4 — fusion 비교

- A: 7→3 input adapter
- B: 페이지 비율 2개를 포함한 9→3 adapter
- C: late feature fusion
- D: gated multi-level fusion
- E: cross-attention fusion

동일 확장 데이터 binary split에서 새로 비교해야 한다. 과거 11-class 결과는 현재 비교표가 아니다.

### 권장 5 — 통계적 안정성

- 최소 3개 seed.
- overall pixel mIoU와 함께 sensor/task/group macro 평균 보고.
- 독립 관리 그룹 수가 적으므로 group bootstrap 또는 leave-one-group-out 검토.
- test는 최종 설정이 고정된 뒤 평가.

### 권장 6 — 실제 배포 관점

- batch 1 FPS 및 peak GPU memory.
- 원본 프레임 stitching 시 경계 artifact.
- threshold calibration은 별도 validation에서 수행.
- 촬영 센서별 성능과 missing-spectrum 동작.

---

## 15. 발표 슬라이드 구성 예시

### 슬라이드 1 — 문제 정의

- 하천 영상에서 algae-related 영역을 픽셀 단위 검출.
- 세부 단계 분류 실패를 binary 문제로 재정의.
- 양성 정의를 표로 명시.

### 슬라이드 2 — 데이터

- 확장본 3,986장, 라벨 사용 3,985장, 58 task.
- Altum/P1 수량.
- 기존 795장 포함 여부 SHA256 검증.

### 슬라이드 3 — 분할

- 프레임 무작위 분할이 아님.
- task + P1 GPS 1km 연결 그룹.
- train/val/test 표와 Altum 메타데이터 한계.

### 슬라이드 4 — 입력

- 9-slot 표.
- 현재 input adapter는 실제 7슬롯 사용.
- P1 missing channels=0, Altum page 의미 미확인.

### 슬라이드 5 — 모델

- 7→3 adapter → DINOv3 ViT-L → Mask2Former.
- CLS-aux branch.
- 약 340.35M parameters.

### 슬라이드 6 — 학습 설정

- crop, batch, accumulation, LR, epoch, validation 및 early stopping.
- loss 구성.

### 슬라이드 7 — 현재 결과

- epoch 곡선 표.
- epoch 30 overall와 클래스별 IoU.
- “validation 중간 결과, test 아님” 표시.

### 슬라이드 8 — 센서·task별 결과

- Altum 98.00%, P1 94.49%.
- task76 25.91%를 숨기지 않고 failure case로 제시.

### 슬라이드 9 — 한계

- Altum 장소 독립성 미확인.
- 라벨 의미 미확인.
- RGB-only 대조군 없음.
- 독립 그룹 수 제한.

### 슬라이드 10 — 다음 실험

- task76 분석 → RGB-only → label ablation → A–E fusion → multi-seed → 최종 test.

---

## 16. 발표 마지막 한 문장

> 현재 결과는 확장 데이터의 binary algae-related segmentation 가능성을 보여주는 validation 중간 결과입니다. 하지만 task별 편차와 센서·라벨 정의의 불확실성이 남아 있으므로, RGB-only 대조군과 실패 장면 분석을 완료한 뒤 최종 test 성능을 보고하겠습니다.

---

## 17. 재현 파일과 명령어

### 핵심 파일

- 데이터 manifest: `data/labeling_binary_v2_geo/manifest.json`
- 분할 검증: `data/labeling_binary_v2_geo/verification.json`
- 학습 설정: `config/labeling_binary_algae_v2.yaml`
- binary 라벨 로더: `goose_semseg/data/binary_algae.py`
- 멀티채널 로더: `goose_semseg/data/spectral_tiles.py`
- fusion 구현: `goose_semseg/models/builder.py`
- 학습 코드: `train.py`
- 실행기: `tools/run_binary_algae.py`
- 로그: `outputs/labeling_binary_algae_v2/train.log`
- epoch 결과: `outputs/labeling_binary_algae_v2/binary_aux005_seed42/epoch_metrics.csv`
- epoch 30 그룹 결과: `outputs/labeling_binary_algae_v2/binary_aux005_seed42/val_groups_epoch_030.json`

### 진행 확인

```bash
tail -f outputs/labeling_binary_algae_v2/train.log
systemctl --user status dummdumm-binary-algae-v2.service --no-pager
```

### 현재 실험 식별 정보

- manifest SHA256: `c789b6a7356573bf6b3035f71e3862a409f14690823738bd636087b0396229b5`
- config SHA256 (smoke 시점): `c40799849c40aa4957d69b58f5a606265017ffbdc5835dfdf6e54553488fd32b`
- random seed: 42
- best checkpoint as of epoch 30: `best_epoch_030_miou_0.9468.pt`
- test inference for current model: 아직 수행하지 않음

---

## 18. 랩미팅 직전 체크리스트

- [ ] 학습이 완료됐는지 또는 몇 epoch까지 진행됐는지 갱신한다.
- [ ] best validation checkpoint와 epoch를 갱신한다.
- [ ] 최종 epoch 곡선을 그래프로 만든다.
- [ ] task76의 원본·정답·예측 시각화를 준비한다.
- [ ] 전체·센서별·task별 confusion matrix를 준비한다.
- [ ] `algae0`, `nps`, `nps_algae`의 공식 라벨링 가이드를 확보한다.
- [ ] Altum TIFF page0~5의 밴드 이름·중심 파장·단위와 page6 생성 경위를 확보한다.
- [ ] RGB-only 대조군의 실행 여부를 명확히 표시한다.
- [ ] test를 평가했다면 평가 횟수와 threshold/TTA 조건을 기록한다.
- [ ] 최종 숫자마다 train/validation/test 중 어느 split인지 표기한다.
