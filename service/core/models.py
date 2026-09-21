"""학습된 모델(.pkl)과 평가 지표(.json) 로딩 및 예측 헬퍼."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd

from ..config import EVAL_DIR, FEATURE_COLS, MODEL_DIR, MODEL_LABELS


@dataclass
class LoadedModel:
    key: str            # 파일명 접두사 (xgboost, lgbm, ...)
    label: str          # 화면 표시명
    estimator: object   # sklearn 호환 분류기
    metrics: dict = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """고장(1) 확률 반환."""
        X = X[FEATURE_COLS].astype(float)
        proba = self.estimator.predict_proba(X)
        return np.asarray(proba)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


def _load_metrics(key: str) -> dict:
    """train/model/eval/data/{key}_eval.json 의 평가 지표."""
    path = EVAL_DIR / f"{key}_eval.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def available_model_keys() -> list[str]:
    """models 폴더에 실제로 저장된 모델 키 목록 (표시 순서 고정)."""
    keys = {p.name.replace("_model.pkl", "") for p in MODEL_DIR.glob("*_model.pkl")}
    ordered = [k for k in MODEL_LABELS if k in keys]
    return ordered + sorted(keys - set(ordered))


def load_model(key: str) -> LoadedModel:
    """pkl 하나를 로드한다. train.py 가 (model, eval) 튜플로 저장하는 경우도 처리."""
    obj = joblib.load(MODEL_DIR / f"{key}_model.pkl")
    metrics = {}
    if isinstance(obj, (tuple, list)):
        estimator = obj[0]
        if len(obj) > 1 and isinstance(obj[1], dict):
            metrics = obj[1]
    else:
        estimator = obj

    # 평가 지표는 eval/data 의 json 을 우선 사용한다.
    metrics = _load_metrics(key) or metrics
    return LoadedModel(
        key=key,
        label=MODEL_LABELS.get(key, key),
        estimator=estimator,
        metrics=metrics,
    )


def load_all_models() -> dict[str, LoadedModel]:
    models = {}
    for key in available_model_keys():
        try:
            models[key] = load_model(key)
        except Exception as exc:  # 모델 버전 불일치 등은 건너뛰고 나머지를 사용
            print(f"[service] {key} 모델 로드 실패: {exc}")
    return models


def metrics_table(models: dict[str, LoadedModel]) -> pd.DataFrame:
    """eval json 기준 모델 성능 비교표."""
    rows = []
    for m in models.values():
        rows.append(
            {
                "모델": m.label,
                "정확도": m.metrics.get("accuracy"),
                "재현율": m.metrics.get("recall"),
                "정밀도": m.metrics.get("precision"),
                "F1": m.metrics.get("f1_score"),
            }
        )
    return pd.DataFrame(rows)
