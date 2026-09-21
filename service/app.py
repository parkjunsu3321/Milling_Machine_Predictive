"""밀링 기계 고장 예측 서비스 (Streamlit).

실행: streamlit run service/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# streamlit run 으로 실행될 때도 service 패키지를 import 할 수 있게 경로 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from service.config import MODEL_DIR, TEST_DATA_PATH  # noqa: E402
from service.core.models import metrics_table  # noqa: E402
from service.views import batch_test, realtime, single_input  # noqa: E402
from service.views.common import get_models  # noqa: E402

st.set_page_config(
    page_title="밀링 기계 고장 예측",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = {
    "개요 · 모델 성능": "overview",
    "실제 데이터 입력 예측": "single",
    "테스트 데이터 예측": "batch",
    "실시간 모니터링 (5대)": "realtime",
}


def overview():
    st.header("밀링 기계 예지보전 서비스")
    st.caption(
        "AI4I 2020 예지보전 데이터로 학습한 모델을 사용해 밀링 기계의 고장을 예측합니다."
    )

    models = get_models()
    table = metrics_table(models)

    st.subheader("학습된 모델 성능")
    st.caption("`train/model/eval/data/*_eval.json` 에 저장된 학습 시점 평가 지표입니다.")
    c1, c2 = st.columns([1.1, 1.4])
    with c1:
        st.dataframe(
            table.style.format({c: "{:.4f}" for c in ["정확도", "재현율", "정밀도", "F1"]})
            .highlight_max(
                subset=["정확도", "재현율", "정밀도", "F1"],
                props="background-color:#111827;color:white;font-weight:700",
            ),
            hide_index=True,
            use_container_width=True,
        )
        best = table.loc[table["F1"].idxmax(), "모델"] if not table["F1"].isna().all() else "-"
        st.success(f"F1 기준 최고 성능 모델: **{best}**")
    with c2:
        melted = table.melt(id_vars="모델", var_name="지표", value_name="값").dropna()
        fig = px.bar(melted, x="지표", y="값", color="모델", barmode="group", range_y=[0.8, 0.95])
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("화면 안내")
    c1, c2, c3 = st.columns(3)
    with c1.container(border=True):
        st.markdown("#### 실제 데이터 입력 예측")
        st.write(
            "주변온도·공정온도·회전속도·토크·공구마모·제품등급을 입력하면 "
            "학습 때와 같은 파생 변수를 계산해 고장 확률을 예측합니다."
        )
    with c2.container(border=True):
        st.markdown("#### 테스트 데이터 예측")
        st.write(
            "저장된 테스트셋이나 직접 올린 CSV 전체를 일괄 예측하고, "
            "정답 라벨이 있으면 정확도·재현율·혼동 행렬까지 계산합니다."
        )
    with c3.container(border=True):
        st.markdown("#### 실시간 모니터링")
        st.write(
            "밀링 기계 5대의 센서 값을 주기적으로 생성해 설비별 고장 확률과 "
            "경보 로그를 실시간으로 보여줍니다."
        )

    with st.expander("데이터·모델 경로"):
        st.code(
            f"모델 폴더 : {MODEL_DIR}\n"
            f"테스트셋  : {TEST_DATA_PATH}\n"
            f"사용 모델 : {', '.join(m.label for m in models.values())}",
            language="text",
        )


def main():
    st.sidebar.title("🛠️ 밀링 고장 예측")
    choice = st.sidebar.radio("화면", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()

    page = PAGES[choice]
    if page == "overview":
        overview()
    elif page == "single":
        single_input.render()
    elif page == "batch":
        batch_test.render()
    else:
        realtime.render()


main()
