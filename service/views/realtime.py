"""5대 밀링 기계 실시간 고장 예측 모니터링 화면."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..config import GRADE_LABELS, risk_level
from ..core.features import build_features
from ..core.simulator import WEAR_LIMIT, MachineFleet
from .common import get_feature_stats, legend_html, model_sidebar


def _fleet() -> MachineFleet:
    if "fleet" not in st.session_state:
        st.session_state.fleet = MachineFleet()
    return st.session_state.fleet


def _predict_step(model, threshold: float, stress_rate: float, load_scale: float) -> pd.DataFrame:
    fleet = _fleet()
    snapshot = fleet.step(stress_rate=stress_rate, load_scale=load_scale)
    proba = model.predict_proba(build_features(snapshot, get_feature_stats()))
    snapshot["고장확률"] = proba
    snapshot["고장예측"] = (proba >= threshold).astype(int)
    snapshot["모델"] = model.label
    fleet.record(snapshot)
    st.session_state.latest = snapshot
    return snapshot


def _machine_card(col, row: dict, threshold: float):
    name, color = risk_level(row["고장확률"])
    with col.container(border=True):
        st.markdown(
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<span style='font-weight:700;font-size:1.05rem'>{row['설비명']}</span>"
            f"<span style='background:{color};color:white;padding:2px 10px;border-radius:999px;"
            f"font-size:0.75rem;font-weight:600'>{name}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='font-size:1.9rem;font-weight:700;color:{color};line-height:1.6'>"
            f"{row['고장확률']*100:.1f}%</div>"
            f"<div style='color:#6b7280;font-size:0.78rem;margin-top:-6px'>고장 확률 "
            f"(임계값 {threshold:.2f})</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(float(row["고장확률"]), 1.0))

        st.markdown(
            f"<div style='font-size:0.82rem;line-height:1.7'>"
            f"토크 <b>{row['토크_Nm']:.1f}</b> Nm<br>"
            f"회전속도 <b>{row['회전속도_rpm']:,.0f}</b> rpm<br>"
            f"온도차 <b>{row['공정온도_K'] - row['주변온도_K']:.2f}</b> K<br>"
            f"공구마모 <b>{row['공구마모_분']:.0f}</b> / {WEAR_LIMIT:.0f} 분"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.progress(min(float(row["공구마모_분"]) / WEAR_LIMIT, 1.0))

        tags = []
        if row.get("고부하"):
            tags.append("고부하 구간")
        if row.get("공구교체"):
            tags.append("공구 교체 완료")
        if row["고장예측"]:
            tags.append("정비 필요")
        st.caption(
            f"{GRADE_LABELS[int(row['제품등급'])]} · " + (" · ".join(tags) if tags else "이상 없음")
        )


def _dashboard(model, threshold: float, stress_rate: float, load_scale: float, advance: bool):
    if advance or "latest" not in st.session_state:
        snapshot = _predict_step(model, threshold, stress_rate, load_scale)
    else:
        snapshot = st.session_state.latest

    fleet = _fleet()
    rows = snapshot.to_dict("records")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("가동 설비", f"{len(rows)} 대")
    c2.metric("고장 경보", f"{int(snapshot['고장예측'].sum())} 대")
    c3.metric("최고 고장 확률", f"{snapshot['고장확률'].max():.1%}")
    c4.metric("누적 틱", f"{fleet.tick}")

    st.markdown(legend_html(), unsafe_allow_html=True)
    st.write("")

    cols = st.columns(len(rows))
    for col, row in zip(cols, rows):
        _machine_card(col, row, threshold)

    hist = fleet.history_df()
    if not hist.empty and hist["시각"].nunique() > 1:
        st.subheader("고장 확률 추이")
        fig = px.line(
            hist.sort_values("시각"),
            x="시각", y="고장확률", color="설비명", markers=False, range_y=[0, 1],
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"임계값 {threshold:.2f}")
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("센서 값 추이 보기"):
            sensor = st.selectbox(
                "센서", ["토크_Nm", "회전속도_rpm", "공구마모_분", "공정온도_K", "주변온도_K"],
                key="rt_sensor",
            )
            fig2 = px.line(hist.sort_values("시각"), x="시각", y=sensor, color="설비명")
            fig2.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
            st.plotly_chart(fig2, use_container_width=True)

    alerts = fleet.alerts_df()
    st.subheader(f"고장 경보 로그 ({len(alerts)}건)")
    if alerts.empty:
        st.success("아직 경보가 없습니다.")
    else:
        show = alerts.head(50).copy()
        show["시각"] = pd.to_datetime(show["시각"]).dt.strftime("%H:%M:%S")
        st.dataframe(
            show.style.format({"고장확률": "{:.3f}", "토크_Nm": "{:.1f}",
                               "회전속도_rpm": "{:.0f}", "공구마모_분": "{:.0f}"}),
            hide_index=True, use_container_width=True, height=260,
        )
        st.download_button(
            "경보 로그 CSV 내려받기",
            alerts.to_csv(index=False).encode("utf-8-sig"),
            file_name="fleet_alerts.csv",
            mime="text/csv",
        )

    st.caption(f"최근 갱신: {pd.Timestamp.now():%Y-%m-%d %H:%M:%S} · 예측 모델: {model.label}")


def render():
    st.header("5대 밀링 기계 실시간 모니터링")
    st.caption(
        "가상 설비 5대의 센서 값을 주기적으로 생성해 선택한 모델로 고장을 실시간 예측합니다."
    )

    model, models, threshold = model_sidebar("rt_")
    fleet = _fleet()

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    running = c1.toggle("자동 실행", value=True, key="rt_running")
    interval = c2.slider("갱신 주기 (초)", 1, 10, 2, key="rt_interval")
    stress_rate = c3.slider("고부하 발생률", 0.0, 0.5, 0.08, 0.02, key="rt_stress")
    load_scale = c4.slider("부하 배율", 0.8, 1.5, 1.0, 0.05, key="rt_load")

    b1, b2, b3 = st.columns([1, 1, 2])
    manual = b1.button("한 틱 진행", disabled=running, use_container_width=True)
    if b2.button("초기화", use_container_width=True):
        st.session_state.fleet = MachineFleet()
        st.session_state.pop("latest", None)
        st.rerun()
    with b3.popover("공구 교체", use_container_width=True):
        target = st.selectbox("대상 설비", list(fleet.machines), key="rt_tool_target")
        if st.button("교체 실행", key="rt_tool_go"):
            fleet.machines[target].replace_tool()
            st.success(f"{target} 공구를 교체했습니다.")

    st.divider()

    if running and hasattr(st, "fragment"):
        fragment = st.fragment(run_every=f"{interval}s")(
            lambda: _dashboard(model, threshold, stress_rate, load_scale, advance=True)
        )
        fragment()
    elif running:
        # 구버전 Streamlit 대비: 한 틱만 진행하고 수동 갱신 안내
        _dashboard(model, threshold, stress_rate, load_scale, advance=True)
        st.info("이 Streamlit 버전은 자동 갱신을 지원하지 않습니다. '한 틱 진행' 버튼을 사용하세요.")
    else:
        _dashboard(model, threshold, stress_rate, load_scale, advance=manual)
