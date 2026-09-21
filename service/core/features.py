"""원본 센서 값 -> 학습에 사용한 파생 변수 재현.

EDA/EDA.ipynb 의 파생 변수 생성 과정을 그대로 옮긴 모듈이다.
MinMaxScaler / z-score / 분위수 기준값은 원본 데이터(ai4i2020.csv) 전체에서
한 번 계산해 artifacts/feature_stats.json 에 저장해두고 추론 때 재사용한다.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ..config import (
    FEATURE_COLS,
    FEATURE_STATS_PATH,
    GRADE_MAP,
    RAW_DATA_PATH,
    RAW_RENAME_MAP,
    TARGET_COL,
)


def load_raw_dataframe() -> pd.DataFrame:
    """ai4i2020.csv 를 한글 컬럼명으로 읽어온다."""
    df = pd.read_csv(RAW_DATA_PATH, encoding="utf-8-sig")
    df = df.rename(columns=RAW_RENAME_MAP)
    df["제품등급"] = df["제품등급"].map(GRADE_MAP).astype(int)
    return df


def _base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """파생 변수 계산에 필요한 1차 변수까지만 만든 프레임."""
    out = pd.DataFrame(index=df.index)
    out["제품등급"] = pd.to_numeric(df["제품등급"], errors="coerce").astype(float)
    out["주변온도_K"] = df["주변온도_K"].astype(float)
    out["공정온도_K"] = df["공정온도_K"].astype(float)
    out["회전속도_rpm"] = df["회전속도_rpm"].astype(float)
    out["토크_Nm"] = df["토크_Nm"].astype(float)
    out["공구마모_분"] = df["공구마모_분"].astype(float)

    out["마모"] = out["토크_Nm"] * out["회전속도_rpm"] * 2 * math.pi / 60
    out["고사용"] = out["토크_Nm"] * out["회전속도_rpm"]
    out["회전속도_파생2차"] = out["회전속도_rpm"] ** 2
    out["토크_2차"] = out["토크_Nm"] ** 2
    out["토크_3차"] = out["토크_Nm"] ** 3
    out["온도차"] = out["공정온도_K"] - out["주변온도_K"]
    return out


def compute_stats(df_raw: pd.DataFrame | None = None) -> dict:
    """원본 데이터에서 스케일러/분위수/z-score 기준값을 계산한다."""
    if df_raw is None:
        df_raw = load_raw_dataframe()

    base = _base_frame(df_raw)

    wear_min, wear_max = float(base["마모"].min()), float(base["마모"].max())
    tool_min, tool_max = float(base["공구마모_분"].min()), float(base["공구마모_분"].max())

    마모_지수 = np.exp((base["마모"] - wear_min) / (wear_max - wear_min))
    마모_토크 = 마모_지수 * base["토크_2차"]
    마모_지수_3차 = 마모_지수 ** 3
    고장위험도 = 마모_토크 * base["공구마모_분"]

    def _z(series: pd.Series) -> dict:
        return {"mean": float(series.mean()), "std": float(series.std(ddof=0))}

    return {
        "minmax": {
            "마모": {"min": wear_min, "max": wear_max},
            "공구마모_분": {"min": tool_min, "max": tool_max},
        },
        "quantile": {
            "마모_토크_95": float(마모_토크.quantile(0.95)),
            "마모_토크_97": float(마모_토크.quantile(0.97)),
            "마모_지수_3차_95": float(마모_지수_3차.quantile(0.95)),
            "고장위험도_95": float(고장위험도.quantile(0.95)),
            "고장위험도_97": float(고장위험도.quantile(0.97)),
        },
        "zscore": {
            "마모_토크": _z(마모_토크),
            "공구마모_분": _z(base["공구마모_분"]),
            "토크_2차": _z(base["토크_2차"]),
            "토크_3차": _z(base["토크_3차"]),
        },
        "ranges": {
            col: {
                "min": float(base[col].min()),
                "max": float(base[col].max()),
                "mean": float(base[col].mean()),
            }
            for col in ["주변온도_K", "공정온도_K", "회전속도_rpm", "토크_Nm", "공구마모_분"]
        },
    }


def get_stats(refresh: bool = False) -> dict:
    """저장된 기준값을 읽고, 없으면 원본 데이터로 새로 만들어 저장한다."""
    if not refresh and FEATURE_STATS_PATH.exists():
        with open(FEATURE_STATS_PATH, encoding="utf-8") as f:
            return json.load(f)

    stats = compute_stats()
    FEATURE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def build_features(df_raw: pd.DataFrame, stats: dict | None = None) -> pd.DataFrame:
    """원본 센서 값 -> 모델 입력 피처(FEATURE_COLS 순서)."""
    stats = stats or get_stats()
    base = _base_frame(df_raw)
    mm = stats["minmax"]
    qt = stats["quantile"]
    zs = stats["zscore"]

    def _scaled(series: pd.Series, key: str) -> pd.Series:
        lo, hi = mm[key]["min"], mm[key]["max"]
        return (series - lo) / (hi - lo)

    def _z(series: pd.Series, key: str) -> pd.Series:
        return (series - zs[key]["mean"]) / zs[key]["std"]

    f = base.copy()
    f["마모_지수"] = np.exp(_scaled(f["마모"], "마모"))
    f["공구마모_지수"] = np.exp(_scaled(f["공구마모_분"], "공구마모_분"))
    f["온도차_로그"] = np.log(f["온도차"].clip(lower=1e-6))

    f["마모_토크"] = f["마모_지수"] * f["토크_2차"]
    f["기계적_스트레스"] = f["온도차"] * f["토크_2차"]
    f["power_low"] = f["회전속도_파생2차"] * f["토크_2차"]
    f["누적_피로"] = f["고사용"] * f["마모_지수"]
    f["마모_지수_2차"] = f["마모_지수"] ** 2
    f["마모_지수_3차"] = f["마모_지수"] ** 3
    f["안전_마진_지수"] = f["토크_Nm"] / (f["회전속도_rpm"] + 1)
    f["노후도"] = (f["마모_지수"] + f["토크_2차"] + f["고사용"]) / 3
    f["고장위험도"] = f["마모_토크"] * f["공구마모_분"]
    f["복합_스트레스"] = f["마모_토크"] * f["온도차"]

    f["마모_토크_극단위험_95"] = (f["마모_토크"] > qt["마모_토크_95"]).astype(int)
    f["마모_지수_3차_극단위험_95"] = (f["마모_지수_3차"] > qt["마모_지수_3차_95"]).astype(int)
    f["고장위험도_극단위험_95"] = (f["고장위험도"] > qt["고장위험도_95"]).astype(int)
    f["고장위험도_극단위험_97"] = (f["고장위험도"] > qt["고장위험도_97"]).astype(int)
    f["마모_토크_극단위험_97"] = (f["마모_토크"] > qt["마모_토크_97"]).astype(int)

    f["종합_위험_스코어"] = f[
        ["마모_토크_극단위험_95", "마모_지수_3차_극단위험_95", "고장위험도_극단위험_95"]
    ].sum(axis=1)

    f["종합_Z_Score"] = (
        _z(f["마모_토크"], "마모_토크")
        + _z(f["공구마모_분"], "공구마모_분")
        + _z(f["토크_2차"], "토크_2차")
    )
    f["종합_Z_Score2"] = (
        _z(f["마모_토크"], "마모_토크")
        + _z(f["공구마모_분"], "공구마모_분")
        + _z(f["토크_3차"], "토크_3차")
    )

    f["고장위험도_2차"] = f["고장위험도"] ** 2
    f["고장위험도_3차"] = f["고장위험도"] ** 3

    return f[FEATURE_COLS].astype(float)


def is_raw_format(df: pd.DataFrame) -> bool:
    """업로드된 CSV 가 원본 센서 형식인지 판별."""
    cols = set(df.columns)
    kor = {"주변온도_K", "공정온도_K", "회전속도_rpm", "토크_Nm", "공구마모_분"}
    eng = set(RAW_RENAME_MAP.keys()) - {"Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"}
    return kor.issubset(cols) or {"Air temperature [K]", "Torque [Nm]"}.issubset(cols) or eng.issubset(cols)


def normalize_raw(df: pd.DataFrame) -> pd.DataFrame:
    """업로드된 원본 형식 CSV 를 한글 컬럼/숫자 제품등급으로 통일."""
    df = df.rename(columns=RAW_RENAME_MAP).copy()
    if df["제품등급"].dtype == object:
        df["제품등급"] = df["제품등급"].map(GRADE_MAP)
    df["제품등급"] = pd.to_numeric(df["제품등급"], errors="coerce").fillna(1).astype(int)
    return df


def prepare_input(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, str]:
    """CSV 한 장을 받아 (모델 입력, 정답 라벨, 형식명) 으로 변환한다."""
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    if set(FEATURE_COLS).issubset(df.columns):
        return df[FEATURE_COLS].astype(float), y, "가공완료(파생변수 포함)"
    if is_raw_format(df):
        raw = normalize_raw(df)
        y = raw[TARGET_COL] if TARGET_COL in raw.columns else None
        return build_features(raw), y, "원본 센서 데이터"
    missing = [c for c in FEATURE_COLS if c not in df.columns][:5]
    raise ValueError(
        "인식할 수 없는 CSV 형식입니다. 원본 센서 컬럼 또는 학습용 파생 변수 컬럼이 필요합니다. "
        f"(예: 누락된 컬럼 {missing} ...)"
    )
