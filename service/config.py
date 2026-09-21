"""서비스 전역 설정: 경로, 모델 레지스트리, 표시용 상수."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SERVICE_DIR = ROOT_DIR / "service"

RAW_DATA_PATH = ROOT_DIR / "data" / "ai4i2020.csv"
TRAIN_DATA_PATH = ROOT_DIR / "train" / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "train" / "data" / "test_data.csv"

MODEL_DIR = ROOT_DIR / "train" / "model" / "models"
EVAL_DIR = ROOT_DIR / "train" / "model" / "eval" / "data"
ARTIFACT_DIR = SERVICE_DIR / "artifacts"
FEATURE_STATS_PATH = ARTIFACT_DIR / "feature_stats.json"

# 파일명 접두사 -> 화면 표시명
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lgbm": "LightGBM",
    "catboost": "CatBoost",
    "tabPFN": "TabPFN",
}

TARGET_COL = "기계고장"

# 사용자가 직접 입력하는 원본 센서 값
RAW_INPUT_COLS = [
    "제품등급",
    "주변온도_K",
    "공정온도_K",
    "회전속도_rpm",
    "토크_Nm",
    "공구마모_분",
]

GRADE_MAP = {"L": 1, "M": 2, "H": 3}
GRADE_LABELS = {1: "L (저품질)", 2: "M (중품질)", 3: "H (고품질)"}

# 원본 Kaggle 컬럼명 -> 한글 컬럼명 (EDA 노트북과 동일)
RAW_RENAME_MAP = {
    "UDI": "고유번호",
    "Product ID": "제품ID",
    "Type": "제품등급",
    "Air temperature [K]": "주변온도_K",
    "Process temperature [K]": "공정온도_K",
    "Rotational speed [rpm]": "회전속도_rpm",
    "Torque [Nm]": "토크_Nm",
    "Tool wear [min]": "공구마모_분",
    "Machine failure": "기계고장",
    "TWF": "공구마모고장",
    "HDF": "열방출고장",
    "PWF": "동력고장",
    "OSF": "과부하고장",
    "RNF": "무작위고장",
}

# 학습 데이터(train_data.csv)의 피처 순서 그대로 유지해야 모델 입력이 맞는다.
FEATURE_COLS = [
    "종합_Z_Score2", "종합_Z_Score", "토크_3차", "마모_토크", "토크_2차",
    "마모_지수_3차", "종합_위험_스코어", "고장위험도", "복합_스트레스",
    "고장위험도_극단위험_95", "고장위험도_극단위험_97", "마모_지수_2차",
    "고장위험도_2차", "power_low", "누적_피로", "안전_마진_지수",
    "기계적_스트레스", "마모_지수", "토크_Nm", "고장위험도_3차", "노후도",
    "제품등급", "고사용", "마모", "온도차_로그", "마모_토크_극단위험_95",
    "온도차", "마모_지수_3차_극단위험_95", "공구마모_지수", "마모_토크_극단위험_97",
]

# 고장 확률 구간별 등급
RISK_LEVELS = [
    (0.30, "정상", "#22c55e"),
    (0.60, "주의", "#eab308"),
    (0.80, "경고", "#f97316"),
    (1.01, "위험", "#ef4444"),
]


def risk_level(prob: float):
    """고장 확률 -> (등급명, 색상)."""
    for upper, name, color in RISK_LEVELS:
        if prob < upper:
            return name, color
    return RISK_LEVELS[-1][1], RISK_LEVELS[-1][2]
