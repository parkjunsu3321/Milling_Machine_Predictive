"""실제 설비 값을 직접 입력해서 고장을 예측하는 화면."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import GRADE_LABELS
from ..core.features import build_features
from .common import gauge, get_feature_stats, legend_html, model_sidebar

PRESETS = {
    "정상 가동": dict(제품등급=2, 주변온도_K=298.5, 공정온도_K=308.6, 회전속도_rpm=1520.0, 토크_Nm=40.0, 공구마모_분=20.0),
    "과부하(고토크·저RPM)": dict(제품등급=1, 주변온도_K=300.5, 공정온도_K=310.8, 회전속도_rpm=1280.0, 토크_Nm=62.0, 공구마모_분=180.0),
    "공구마모 한계": dict(제품등급=3, 주변온도_K=299.0, 공정온도_K=309.4, 회전속도_rpm=1420.0, 토크_Nm=52.0, 공구마모_분=225.0),
    "열방출 위험(온도차 부족)": dict(제품등급=1, 주변온도_K=303.0, 공정온도_K=311.2, 회전속도_rpm=1310.0, 토크_Nm=50.0, 공구마모_분=120.0),
}


def _sensitivity(model, base: dict, col: str, values: np.ndarray) -> pd.DataFrame:
    rows = []
    for v in values:
        row = dict(base)
        row[col] = float(v)
        rows.append(row)
    df = pd.DataFrame(rows)
    proba = model.predict_proba(build_features(df, get_feature_stats()))
    return pd.DataFrame({col: values, "고장확률": proba})


def render():
    st.header("실제 데이터 입력 예측")
    st.caption(
        "현장 센서 값을 입력하면 EDA 단계에서 만든 파생 변수 30개를 동일하게 계산해 모델에 넣습니다."
    )

    model, models, threshold = model_sidebar("single_")
    stats = get_feature_stats()

    if "single_values" not in st.session_state:
        st.session_state.single_values = dict(PRESETS["정상 가동"])

    st.write("**빠른 입력 프리셋**")
    cols = st.columns(len(PRESETS))
    for col, (name, values) in zip(cols, PRESETS.items()):
        if col.button(name, use_container_width=True):
            st.session_state.single_values = dict(values)

    v = st.session_state.single_values
    r = stats["ranges"]

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        grade = c1.selectbox(
            "제품등급",
            [1, 2, 3],
            index=[1, 2, 3].index(int(v["제품등급"])),
            format_func=lambda g: GRADE_LABELS[g],
        )
        air = c1.number_input(
            "주변온도 (K)", 290.0, 315.0, float(v["주변온도_K"]), 0.1,
            help=f"학습 데이터 범위 {r['주변온도_K']['min']:.1f} ~ {r['주변온도_K']['max']:.1f} K",
        )
        proc = c2.number_input(
            "공정온도 (K)", 295.0, 320.0, float(v["공정온도_K"]), 0.1,
            help=f"학습 데이터 범위 {r['공정온도_K']['min']:.1f} ~ {r['공정온도_K']['max']:.1f} K",
        )
        rpm = c2.number_input(
            "회전속도 (rpm)", 1000.0, 3000.0, float(v["회전속도_rpm"]), 10.0,
            help=f"학습 데이터 범위 {r['회전속도_rpm']['min']:.0f} ~ {r['회전속도_rpm']['max']:.0f} rpm",
        )
        torque = c3.number_input(
            "토크 (Nm)", 1.0, 90.0, float(v["토크_Nm"]), 0.5,
            help=f"학습 데이터 범위 {r['토크_Nm']['min']:.1f} ~ {r['토크_Nm']['max']:.1f} Nm",
        )
        wear = c3.number_input(
            "공구마모 (분)", 0.0, 300.0, float(v["공구마모_분"]), 1.0,
            help=f"학습 데이터 범위 {r['공구마모_분']['min']:.0f} ~ {r['공구마모_분']['max']:.0f} 분",
        )
        submitted = st.form_submit_button("고장 예측 실행", type="primary", use_container_width=True)

    if proc <= air:
        st.warning("공정온도는 주변온도보다 높아야 합니다. (온도차 로그 변환 때문)")

    if submitted:
        st.session_state.single_values = dict(
            제품등급=grade, 주변온도_K=air, 공정온도_K=proc,
            회전속도_rpm=rpm, 토크_Nm=torque, 공구마모_분=wear,
        )

    base = st.session_state.single_values
    raw = pd.DataFrame([base])
    features = build_features(raw, stats)
    prob = float(model.predict_proba(features)[0])
    pred = int(prob >= threshold)

    st.divider()
    left, right = st.columns([1, 1.3])
    with left:
        st.plotly_chart(gauge(prob, f"{model.label} 고장 확률"), use_container_width=True)
        st.markdown(legend_html(), unsafe_allow_html=True)
    with right:
        st.subheader("판정 결과")
        if pred:
            st.error(f"고장 예측: **고장 위험** (확률 {prob:.1%}, 임계값 {threshold:.2f})")
        else:
            st.success(f"고장 예측: **정상** (확률 {prob:.1%}, 임계값 {threshold:.2f})")

        power = base["토크_Nm"] * base["회전속도_rpm"] * 2 * np.pi / 60
        c1, c2, c3 = st.columns(3)
        c1.metric("기계 동력", f"{power:,.0f} W")
        c2.metric("온도차", f"{base['공정온도_K'] - base['주변온도_K']:.2f} K")
        c3.metric("마모×토크", f"{base['공구마모_분'] * base['토크_Nm']:,.0f}")

        checks = []
        if not 3500 <= power <= 9000:
            checks.append("동력이 정상 구간(3.5~9kW)을 벗어났습니다 → 동력 고장 위험")
        if base["공정온도_K"] - base["주변온도_K"] < 8.6 and base["회전속도_rpm"] < 1380:
            checks.append("온도차 8.6K 미만 + 회전속도 1380rpm 미만 → 열방출 고장 위험")
        limit = {1: 11000, 2: 12000, 3: 13000}[int(base["제품등급"])]
        if base["공구마모_분"] * base["토크_Nm"] > limit:
            checks.append(f"마모×토크가 {limit:,} 초과 → 과부하 고장 위험")
        if base["공구마모_분"] > 200:
            checks.append("공구마모 200분 초과 → 공구 교체 시점")
        if checks:
            st.warning("**현장 점검 포인트**\n\n- " + "\n- ".join(checks))
        else:
            st.info("주요 물리 조건(동력·온도차·마모×토크)은 모두 정상 범위입니다.")

    st.subheader("모델별 예측 비교")
    compare = pd.DataFrame(
        [
            {
                "모델": m.label,
                "고장확률": float(m.predict_proba(features)[0]),
                "판정": "고장" if float(m.predict_proba(features)[0]) >= threshold else "정상",
                "학습 F1": m.metrics.get("f1_score"),
            }
            for m in models.values()
        ]
    )
    fig = px.bar(
        compare, x="모델", y="고장확률", color="판정", text="고장확률",
        color_discrete_map={"고장": "#ef4444", "정상": "#22c55e"}, range_y=[0, 1],
    )
    fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
    fig.add_hline(y=threshold, line_dash="dash", annotation_text=f"임계값 {threshold:.2f}")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("조건 변화에 따른 고장 확률")
    t1, t2 = st.tabs(["공구마모 변화", "토크 변화"])
    with t1:
        curve = _sensitivity(model, base, "공구마모_분", np.linspace(0, 260, 40))
        line = px.line(curve, x="공구마모_분", y="고장확률", markers=False, range_y=[0, 1])
        line.add_vline(x=base["공구마모_분"], line_dash="dot", annotation_text="현재")
        line.add_hline(y=threshold, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(line, use_container_width=True)
    with t2:
        curve = _sensitivity(model, base, "토크_Nm", np.linspace(5, 80, 40))
        line = px.line(curve, x="토크_Nm", y="고장확률", markers=False, range_y=[0, 1])
        line.add_vline(x=base["토크_Nm"], line_dash="dot", annotation_text="현재")
        line.add_hline(y=threshold, line_dash="dash", line_color="#ef4444")
        st.plotly_chart(line, use_container_width=True)

    with st.expander("모델에 들어간 파생 변수 30개 보기"):
        st.dataframe(features.T.rename(columns={0: "값"}), use_container_width=True)
