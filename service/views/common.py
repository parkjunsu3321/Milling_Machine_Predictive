"""화면 공통 요소: 모델 로딩/선택 사이드바, 게이지, 상태 배지."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..config import RISK_LEVELS, risk_level
from ..core.features import get_stats
from ..core.models import LoadedModel, load_all_models, metrics_table


@st.cache_resource(show_spinner="모델 불러오는 중...")
def get_models() -> dict[str, LoadedModel]:
    return load_all_models()


@st.cache_data(show_spinner=False)
def get_feature_stats() -> dict:
    return get_stats()


def model_sidebar(key_prefix: str = "") -> tuple[LoadedModel, dict[str, LoadedModel], float]:
    """사이드바에 모델 선택 + 성능(eval json) 표시. (선택모델, 전체모델, 임계값) 반환."""
    models = get_models()
    if not models:
        st.sidebar.error("train/model/models 에 학습된 모델이 없습니다.")
        st.stop()

    st.sidebar.subheader("모델 선택")
    keys = list(models)
    labels = {k: models[k].label for k in keys}
    selected = st.sidebar.selectbox(
        "예측에 사용할 모델",
        keys,
        format_func=lambda k: labels[k],
        key=f"{key_prefix}model_select",
    )
    model = models[selected]

    metrics = model.metrics
    if metrics:
        st.sidebar.caption("학습 시 평가 지표 (eval/data/*.json)")
        c1, c2 = st.sidebar.columns(2)
        c1.metric("정확도", f"{metrics.get('accuracy', 0):.3f}")
        c2.metric("재현율", f"{metrics.get('recall', 0):.3f}")
        c3, c4 = st.sidebar.columns(2)
        c3.metric("정밀도", f"{metrics.get('precision', 0):.3f}")
        c4.metric("F1", f"{metrics.get('f1_score', 0):.3f}")
    else:
        st.sidebar.caption("평가 지표 파일이 없습니다.")

    threshold = st.sidebar.slider(
        "고장 판정 임계값", 0.05, 0.95, 0.50, 0.05, key=f"{key_prefix}threshold"
    )

    with st.sidebar.expander("전체 모델 성능 비교"):
        table = metrics_table(models)
        st.dataframe(
            table.style.format({c: "{:.4f}" for c in ["정확도", "재현율", "정밀도", "F1"]}),
            hide_index=True,
            use_container_width=True,
        )

    return model, models, threshold


def gauge(prob: float, title: str = "고장 확률", height: int = 260) -> go.Figure:
    name, color = risk_level(prob)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": f"{title} · {name}", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.75},
                "steps": [
                    {"range": [0, 30], "color": "#dcfce7"},
                    {"range": [30, 60], "color": "#fef9c3"},
                    {"range": [60, 80], "color": "#ffedd5"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
                "threshold": {
                    "line": {"color": "#111827", "width": 3},
                    "thickness": 0.8,
                    "value": prob * 100,
                },
            },
        )
    )
    fig.update_layout(height=height, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def risk_badge(prob: float) -> str:
    name, color = risk_level(prob)
    return (
        f"<span style='background:{color};color:white;padding:2px 10px;"
        f"border-radius:999px;font-size:0.8rem;font-weight:600'>{name} "
        f"{prob*100:.1f}%</span>"
    )


def legend_html() -> str:
    parts = []
    prev = 0.0
    for upper, name, color in RISK_LEVELS:
        hi = min(upper, 1.0)
        parts.append(
            f"<span style='display:inline-block;margin-right:12px'>"
            f"<span style='display:inline-block;width:10px;height:10px;background:{color};"
            f"border-radius:2px;margin-right:4px'></span>{name} "
            f"({prev*100:.0f}~{hi*100:.0f}%)</span>"
        )
        prev = hi
    return "".join(parts)


def style_prob_table(df: pd.DataFrame, prob_col: str = "고장확률"):
    return df.style.background_gradient(cmap="Reds", subset=[prob_col]).format(
        {prob_col: "{:.3f}"}
    )
