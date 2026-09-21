"""테스트 데이터(또는 업로드 CSV)로 고장을 일괄 예측하는 화면."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ..config import RAW_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH
from ..core.features import prepare_input
from .common import get_feature_stats, model_sidebar

SOURCES = {
    "테스트 데이터 (train/data/test_data.csv)": TEST_DATA_PATH,
    "학습 데이터 (train/data/train_data.csv)": TRAIN_DATA_PATH,
    "원본 센서 데이터 (data/ai4i2020.csv)": RAW_DATA_PATH,
}


@st.cache_data(show_spinner=False)
def _load_csv(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str, encoding="utf-8-sig")


def _metric_row(y_true, y_pred, proba) -> dict:
    out = {
        "정확도": accuracy_score(y_true, y_pred),
        "재현율": recall_score(y_true, y_pred, zero_division=0),
        "정밀도": precision_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        out["ROC-AUC"] = roc_auc_score(y_true, proba)
    return out


def render():
    st.header("테스트 데이터 예측")
    st.caption("저장된 데이터셋이나 직접 올린 CSV 전체에 대해 고장을 예측하고 성능을 확인합니다.")

    model, models, threshold = model_sidebar("batch_")
    stats = get_feature_stats()

    source = st.radio(
        "데이터 선택", list(SOURCES) + ["CSV 업로드"], horizontal=True, label_visibility="visible"
    )

    if source == "CSV 업로드":
        uploaded = st.file_uploader(
            "CSV 업로드 (원본 센서 컬럼 또는 학습용 파생 변수 컬럼)", type=["csv"]
        )
        if uploaded is None:
            st.info(
                "업로드 가능한 형식\n\n"
                "1) 원본 센서 형식: `Air temperature [K]`, `Process temperature [K]`, "
                "`Rotational speed [rpm]`, `Torque [Nm]`, `Tool wear [min]`, `Type` "
                "(또는 한글 컬럼명)\n"
                "2) 가공 완료 형식: train_data.csv 와 동일한 30개 파생 변수 컬럼\n\n"
                "`기계고장` 컬럼이 있으면 성능 지표까지 함께 계산합니다."
            )
            template = pd.DataFrame(
                {
                    "제품등급": ["M", "L", "H"],
                    "주변온도_K": [298.1, 300.4, 299.0],
                    "공정온도_K": [308.6, 310.9, 309.3],
                    "회전속도_rpm": [1551, 1320, 1430],
                    "토크_Nm": [42.8, 60.1, 51.0],
                    "공구마모_분": [0, 190, 215],
                }
            )
            st.download_button(
                "입력 양식 CSV 내려받기",
                template.to_csv(index=False).encode("utf-8-sig"),
                file_name="input_template.csv",
                mime="text/csv",
            )
            return
        df = pd.read_csv(uploaded, encoding="utf-8-sig")
    else:
        path = SOURCES[source]
        if not path.exists():
            st.error(f"파일을 찾을 수 없습니다: {path}")
            return
        df = _load_csv(str(path))

    try:
        X, y, kind = prepare_input(df)
    except ValueError as exc:
        st.error(str(exc))
        return

    n = len(X)
    max_rows = st.slider("예측에 사용할 행 수", 100, n, min(n, 5000), 100) if n > 100 else n
    X = X.iloc[:max_rows]
    y = y.iloc[:max_rows] if y is not None else None
    df_view = df.iloc[:max_rows].reset_index(drop=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("행 수", f"{len(X):,}")
    c2.metric("데이터 형식", kind)
    c3.metric("정답 라벨", "있음" if y is not None else "없음")

    with st.spinner("예측 중..."):
        proba = model.predict_proba(X)
    pred = (proba >= threshold).astype(int)

    st.divider()
    st.subheader(f"{model.label} 예측 결과")
    c1, c2, c3 = st.columns(3)
    c1.metric("고장 예측 건수", f"{int(pred.sum()):,}")
    c2.metric("고장 예측 비율", f"{pred.mean():.2%}")
    c3.metric("평균 고장 확률", f"{proba.mean():.3f}")

    if y is not None:
        y_true = y.astype(int).to_numpy()
        m = _metric_row(y_true, pred, proba)
        cols = st.columns(len(m))
        for col, (name, value) in zip(cols, m.items()):
            col.metric(name, f"{value:.4f}")
        st.caption(
            f"사이드바에 표시된 지표는 학습 당시 저장된 `{model.key}_eval.json` 값이고, "
            "위 지표는 지금 선택한 데이터와 임계값으로 다시 계산한 값입니다."
        )

        cm = confusion_matrix(y_true, pred)
        left, right = st.columns(2)
        with left:
            fig = px.imshow(
                cm,
                text_auto=True,
                color_continuous_scale="Blues",
                labels=dict(x="예측", y="실제", color="건수"),
                x=["정상", "고장"],
                y=["정상", "고장"],
                title="혼동 행렬",
            )
            fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        with right:
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, proba)
                prec, rec, _ = precision_recall_curve(y_true, proba)
                curve_kind = st.radio("곡선", ["ROC", "PR"], horizontal=True, key="batch_curve")
                if curve_kind == "ROC":
                    fig = px.area(x=fpr, y=tpr, labels=dict(x="위양성률", y="재현율"), title="ROC 곡선")
                    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
                else:
                    fig = px.area(x=rec, y=prec, labels=dict(x="재현율", y="정밀도"), title="Precision-Recall 곡선")
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("정답 라벨이 한 종류뿐이라 곡선을 그릴 수 없습니다.")

    st.subheader("고장 확률 분포")
    hist_df = pd.DataFrame({"고장확률": proba})
    if y is not None:
        hist_df["실제"] = np.where(y.to_numpy() == 1, "고장", "정상")
        fig = px.histogram(hist_df, x="고장확률", color="실제", nbins=50, barmode="overlay",
                           color_discrete_map={"고장": "#ef4444", "정상": "#22c55e"})
    else:
        fig = px.histogram(hist_df, x="고장확률", nbins=50)
    fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"임계값 {threshold:.2f}")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if len(models) > 1:
        st.subheader("모델별 성능 비교 (선택한 데이터 기준)")
        if st.button("전체 모델로 예측해 비교하기"):
            rows = []
            with st.spinner("모든 모델로 예측 중..."):
                for m in models.values():
                    p = m.predict_proba(X)
                    pr = (p >= threshold).astype(int)
                    row = {"모델": m.label, "고장예측비율": pr.mean()}
                    if y is not None:
                        row.update(_metric_row(y.astype(int).to_numpy(), pr, p))
                    row["학습 F1(eval json)"] = m.metrics.get("f1_score")
                    rows.append(row)
            comp = pd.DataFrame(rows)
            st.dataframe(
                comp.style.format({c: "{:.4f}" for c in comp.columns if c != "모델"}),
                hide_index=True,
                use_container_width=True,
            )

    st.subheader("예측 결과 표")
    result = df_view.copy()
    result.insert(0, "고장확률", proba)
    result.insert(1, "고장예측", pred)
    only_risk = st.checkbox("고장으로 예측된 행만 보기", value=False)
    view = result[result["고장예측"] == 1] if only_risk else result
    view = view.sort_values("고장확률", ascending=False)
    st.dataframe(view.head(300), use_container_width=True, height=380)
    st.caption(f"상위 300행만 표시 중 (전체 {len(view):,}행). 전체 결과는 아래에서 내려받으세요.")
    st.download_button(
        "예측 결과 CSV 내려받기",
        result.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"prediction_{model.key}.csv",
        mime="text/csv",
    )
