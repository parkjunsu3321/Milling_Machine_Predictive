# Milling Machine Predictive Maintenance

밀링 기계의 센서 데이터로 기계 고장 여부를 예측하는 머신러닝 프로젝트입니다. AI4I 2020 데이터를 탐색하고, 센서 간 상호작용을 반영한 피처를 생성한 뒤 **CatBoost·LightGBM·XGBoost·TabPFN**을 비교합니다. 학습 모델은 Streamlit 서비스에서 단건 예측, CSV 일괄 예측, 가상 설비 모니터링에 사용합니다.

- [EDA 노트북](EDA/EDA.ipynb): 데이터 탐색, 파생 변수 생성, SMOTENC, 학습 데이터 저장
- [학습 진입점](train/train.py): 모델 학습 및 모델·평가 지표 저장
- [서비스 사용 안내](service/README.md): 웹 화면, 실행 방법, 입력 데이터 규격

## 프로젝트 구조

```text
Milling_Machine_Predictive/
├── data_loader.py                # Kaggle 데이터 다운로드
├── data/ai4i2020.csv             # 원본 데이터
├── EDA/EDA.ipynb                 # 탐색 및 전처리 실험
├── train/
│   ├── train.py                 # 4개 모델 학습·저장
│   ├── data/
│   │   ├── train_data.csv       # 학습용 데이터
│   │   └── test_data.csv        # 별도 저장된 평가용 데이터
│   └── model/
│       ├── data_loader.py       # CSV 로더
│       ├── fit/                 # 모델별 학습 설정
│       ├── eval/
│       │   ├── eval_metric.py   # 분류 평가 함수
│       │   └── data/            # 모델별 평가 JSON
│       └── models/              # 저장된 모델 PKL
└── service/                     # Streamlit 예측 서비스
```

## 데이터

다운로드 스크립트가 사용하는 Kaggle 데이터셋 식별자는 `stephanmatzka/predictive-maintenance-dataset-ai4i-2020`입니다. 저장소의 원본 CSV는 **10,000행·14열**이며, 타깃은 `Machine failure`를 한글로 변환한 `기계고장`입니다. 정상은 0, 고장은 1로 표시합니다.

| 원본 항목 | 전처리 후 이름 | 의미·단위 |
| --- | --- | --- |
| Type | 제품등급 | L → 1, M → 2, H → 3 |
| Air temperature [K] | 주변온도_K | 주변 온도, K |
| Process temperature [K] | 공정온도_K | 공정 온도, K |
| Rotational speed [rpm] | 회전속도_rpm | 회전속도, rpm |
| Torque [Nm] | 토크_Nm | 토크, Nm |
| Tool wear [min] | 공구마모_분 | 공구마모 시간, 분 |
| Machine failure | 기계고장 | 이진 분류 타깃 |

원본은 정상 **9,661건(96.61%)**, 고장 **339건(3.39%)**으로 불균형합니다. 고장 유형 열(TWF, HDF, PWF, OSF, RNF)과 고유번호는 피처 생성 대상에서 제거하며, 제품ID도 학습 입력에서 제외합니다.

## EDA와 피처 엔지니어링

[EDA/EDA.ipynb](EDA/EDA.ipynb)에서 다음 작업을 수행합니다.

1. 컬럼명을 한글로 바꾸고 제품등급을 숫자로 매핑합니다.
2. 클래스 분포와 상관관계를 확인하고, RPM·토크·공구마모·온도차 구간별 고장률을 시각화합니다.
3. 다항식·지수·로그 변환과 센서 간 결합 변수를 생성합니다.
4. SMOTENC로 학습 부분의 소수 클래스를 증강합니다.
5. 증강 데이터를 다시 분할하고, 학습 부분에서 타깃과의 절대 상관계수 상위 **30개 피처**를 선택해 CSV로 저장합니다. 일부 주석에는 20개라고 적혀 있으나 실제 코드는 `head(30)`입니다.

대표적인 피처는 다음과 같습니다.

| 피처 | 계산 방법 |
| --- | --- |
| 마모 | 토크 × RPM × 2π / 60; 이름은 마모지만 계산식은 회전 동력에 해당 |
| 마모_지수 | exp(MinMax(마모)) |
| 토크_2차 / 토크_3차 | 토크의 제곱 / 세제곱 |
| 공구마모_지수 | exp(MinMax(공구마모_분)) |
| 온도차 / 온도차_로그 | 공정온도 − 주변온도 / 해당 값의 자연로그 |
| 마모_토크 | 마모_지수 × 토크_2차 |
| 고장위험도 | 마모_토크 × 공구마모_분 |
| 종합_Z_Score | 마모_토크, 공구마모_분, 토크_2차의 Z-score 합 |
| 종합_Z_Score2 | 위 구성에서 토크_2차를 토크_3차로 대체 |
| 극단위험 플래그 | 해당 변수의 95% 또는 97% 분위수 초과 여부 |

최종 입력 피처와 순서는 [service/config.py](service/config.py)의 `FEATURE_COLS`와 저장된 학습 CSV 헤더에서 확인할 수 있습니다.

### 데이터 분할 과정

현재 노트북과 학습 코드의 실제 흐름은 다음과 같습니다.

```text
원본 10,000행
  → 80:20 계층 분할 (random_state=42)
  → 학습 8,000행에 SMOTENC 적용
  → 증강된 15,458행을 다시 80:20 계층 분할
      ├── train_data.csv: 12,366행
      └── test_data.csv:   3,092행
  → 각 모델은 train_data.csv를 다시 75:25 계층 분할하여 학습·평가
     (random_state=1)
```

| 저장 파일 | 행 수 | 정상 | 고장 | 열 수 |
| --- | ---: | ---: | ---: | ---: |
| train/data/train_data.csv | 12,366 | 6,183 | 6,183 | 31 |
| train/data/test_data.csv | 3,092 | 1,546 | 1,546 | 31 |

31개 열은 입력 피처 30개와 타깃 1개입니다. 최초 분할에서 남긴 원본 20%는 현재 노트북에서 별도 파일로 저장하지 않습니다. `train/train.py`도 `test_data.csv`를 사용하지 않으며, 저장되는 평가 지표는 `train_data.csv` 내부 분할에 대한 결과입니다.

## 모델 학습 및 저장 결과

| 모델 | 주요 설정 |
| --- | --- |
| CatBoost | iterations=200, learning_rate=0.2, depth=1, random_seed=42 |
| LightGBM | n_estimators=200, learning_rate=0.2, max_depth=1, random_state=42 |
| XGBoost | n_estimators=200, learning_rate=0.2, max_depth=1, random_state=42 |
| TabPFN | n_estimators=1, device='cuda', n_preprocessing_jobs=4, ignore_pretraining_limits=True, random_state=42 |

아래 값은 [train/model/eval/data](train/model/eval/data)에 저장된 JSON을 소수점 넷째 자리로 반올림한 값입니다. README 작성 과정에서 재학습한 결과는 아닙니다.

| 모델 | Accuracy | Recall | Precision | F1 |
| --- | ---: | ---: | ---: | ---: |
| CatBoost | 0.8942 | 0.8836 | 0.9028 | 0.8931 |
| LightGBM | 0.8949 | 0.8797 | 0.9073 | 0.8933 |
| XGBoost | 0.8923 | 0.8745 | 0.9068 | 0.8904 |
| TabPFN | 0.9987 | 0.9974 | 1.0000 | 0.9987 |

저장된 지표에서는 TabPFN의 F1이 가장 높습니다. 다만 스케일·분위수·Z-score 기준값을 원본 전체에서 계산하고, SMOTENC 증강 후 다시 학습·평가 데이터를 나누므로 평가 독립성이 제한됩니다. 이 결과를 원본 분포의 미관측 설비 성능으로 해석할 수는 없습니다. 독립적인 성능 평가에는 원본 홀드아웃을 유지하고 학습 부분에만 전처리 기준값 학습·증강·피처 선택을 적용하는 구성이 필요합니다.

학습 완료 후 다음 산출물이 저장됩니다.

- `train/model/models/{모델명}_model.pkl`: `(학습 모델, 평가 지표 dict)` 튜플을 joblib으로 저장
- `train/model/eval/data/{모델명}_eval.json`: Accuracy, Recall, Precision, F1
- 파일명 접두사: `catboost`, `lgbm`, `xgboost`, `tabPFN`

## 실행 방법

아래 명령은 프로젝트 루트에서 실행합니다. 전체 학습에는 현재 TabPFN 설정에 맞는 CUDA 지원 GPU와 PyTorch 환경이 필요합니다. 패키지 버전을 고정한 학습용 의존성 파일은 없으므로, 아래는 코드의 import를 기준으로 한 설치 목록입니다.

### 1. 환경 준비

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r service/requirements.txt
.\.venv\Scripts\python.exe -m pip install jupyterlab matplotlib seaborn scipy imbalanced-learn kagglehub python-dotenv tabpfn
```

TabPFN 학습 함수는 `load_dotenv()`를 호출합니다. 모델 다운로드나 인증이 필요한 환경에서는 해당 환경의 설정을 준비해야 합니다. GPU 지원 여부는 설치한 PyTorch에서 확인합니다.

```powershell
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### 2. 원본 데이터 준비 및 EDA

원본 CSV가 없다면 다운로드 스크립트를 실행합니다.

```powershell
.\.venv\Scripts\python.exe data_loader.py
.\.venv\Scripts\python.exe -m jupyterlab
```

JupyterLab에서 `EDA/EDA.ipynb`를 열고 위에서 아래로 실행합니다. 노트북은 `../data`, `../train/data`를 사용하므로 커널 작업 디렉터리를 `EDA/`로 맞춥니다. 마지막 셀은 기존 학습·테스트 CSV를 덮어씁니다. 저장된 CSV로 학습만 하려면 이 단계는 생략할 수 있습니다.

### 3. 모델 학습

```powershell
.\.venv\Scripts\python.exe -m train.train
```

상대 import를 사용하므로 프로젝트 루트에서 모듈로 실행합니다. 네 모델을 순서대로 학습한 후 저장하며, TabPFN까지 성공해야 저장 루프에 도달합니다. 재실행 시 같은 이름의 모델·평가 파일을 덮어씁니다.

### 4. 서비스 실행

```powershell
.\.venv\Scripts\python.exe -m streamlit run service/app.py
```

실행 후 터미널에 표시되는 주소로 접속합니다. 서비스만 사용할 때의 준비 사항과 CSV 예시는 [service/README.md](service/README.md)를 참고하세요.
