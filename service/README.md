# 밀링 기계 고장 예측 서비스 (Streamlit)

`train/model/models` 에 저장된 학습 모델로 밀링 기계의 고장을 예측하는 웹 서비스입니다.

## 실행

```bash
pip install -r service/requirements.txt
streamlit run service/app.py
```

## 화면 구성

| 화면 | 설명 |
| --- | --- |
| 개요 · 모델 성능 | `train/model/eval/data/*_eval.json` 의 정확도·재현율·정밀도·F1 비교 |
| 실제 데이터 입력 예측 | 센서 값 6개를 직접 입력해 단건 예측, 모델별 비교, 조건 변화에 따른 확률 곡선 |
| 테스트 데이터 예측 | 저장된 데이터셋/업로드 CSV 일괄 예측, 혼동 행렬·ROC·PR, 결과 CSV 저장 |
| 실시간 모니터링 (5대) | 밀링 기계 5대의 센서 값을 주기적으로 생성해 실시간 고장 예측 및 경보 |

모델 선택과 고장 판정 임계값은 모든 화면의 사이드바에서 바꿀 수 있고,
선택한 모델의 학습 성능(eval json)이 사이드바에 함께 표시됩니다.

## 입력 데이터 형식

업로드 CSV 는 두 가지 형식을 지원합니다.

1. **원본 센서 형식** — `Type`, `Air temperature [K]`, `Process temperature [K]`,
   `Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]` (한글 컬럼명도 가능).
   서비스가 EDA 단계의 파생 변수 30개를 동일하게 계산해 모델에 넣습니다.
2. **가공 완료 형식** — `train/data/train_data.csv` 와 같은 파생 변수 30개 컬럼.

`기계고장` 컬럼이 포함되어 있으면 예측 성능 지표까지 계산합니다.

## 구조

```
service/
├── app.py                  # 진입점 + 화면 라우팅
├── config.py               # 경로, 피처 목록, 위험 등급 기준
├── core/
│   ├── features.py         # 원본 센서 값 -> 파생 변수 30개 (EDA 파이프라인 재현)
│   ├── models.py           # pkl 모델 + eval json 로딩, 예측 헬퍼
│   └── simulator.py        # 밀링 기계 5대 센서 시뮬레이터
├── views/
│   ├── common.py           # 모델 선택 사이드바, 게이지, 배지
│   ├── single_input.py     # 실제 데이터 입력 예측
│   ├── batch_test.py       # 테스트 데이터 / CSV 일괄 예측
│   └── realtime.py         # 5대 실시간 모니터링
└── artifacts/
    └── feature_stats.json  # MinMax·z-score·분위수 기준값 (원본 데이터에서 산출)
```

## 파생 변수 기준값

`마모_지수`, `공구마모_지수` 의 MinMax 스케일, `종합_Z_Score` 의 평균·표준편차,
`극단위험` 플래그의 95/97 분위수는 원본 데이터(`data/ai4i2020.csv`) 전체에서
한 번 계산해 `artifacts/feature_stats.json` 에 저장합니다. 파일이 없으면 첫 실행 때
자동으로 생성되며, 데이터가 바뀌면 파일을 지우고 다시 실행하면 재생성됩니다.
