from pathlib import Path

import pandas as pd
import streamlit as st

from modules.data_fetcher import get_recent_data
from modules.predictor import QuantPredictor
from visualization import plot_price_history, plot_sentiment_index


MODEL_FILES = {
    "Model A": Path("models/best_xgboost_panel_model_A.pkl"),
    "Model B": Path("models/best_xgboost_panel_model_B.pkl"),
    "Model C": Path("models/best_xgboost_panel_model_C.pkl"),
    "Scaler": Path("models/scaler.pkl"),
    "PCA": Path("models/pca.pkl"),
}

BACKTEST_DA = {
    "A": 52.94,
    "B": 56.30,
    "C": 55.46,
}


@st.cache_resource
def load_predictor() -> QuantPredictor:
    return QuantPredictor()


def validate_model_files() -> list[str]:
    return [label for label, path in MODEL_FILES.items() if not path.exists()]


def load_input_data(asset_name: str):
    cached = st.session_state.get("raw_data")
    if cached is not None:
        return cached

    df = get_recent_data(asset_name)
    st.session_state.raw_data = df
    return df


def run() -> None:
    st.header("모델 구축 및 검증")
    st.caption("모델 학습은 `model_train.py` 또는 학습 파이프라인에서 수행하고, 대시보드는 저장된 pkl 파일만 사용합니다.")

    missing = validate_model_files()
    if missing:
        st.error(f"필요한 모델 파일이 없습니다: {', '.join(missing)}")
        st.info("먼저 모델 학습 스크립트를 실행해 `models/` 폴더에 pkl 파일을 생성하세요.")
        return

    model_choice = st.radio(
        "사용할 저장 모델",
        options=["A", "B", "C"],
        horizontal=True,
        format_func=lambda value: {
            "A": "Model A (PCA 심리지수)",
            "B": "Model B (잔차 3개)",
            "C": "Model C (PCA + 잔차)",
        }[value],
    )

    asset_name = st.session_state.get("asset_name", "삼성전자")

    if st.button("저장된 모델로 예측 실행", type="primary"):
        with st.spinner(f"{asset_name} 최근 데이터를 준비하고 Model {model_choice}로 예측하는 중입니다."):
            predictor = load_predictor()
            raw_data = load_input_data(asset_name)
            st.session_state.prediction_result = predictor.get_prediction(
                asset_name,
                raw_data,
                model_type=model_choice,
            )
            st.session_state.prediction_model_choice = model_choice

    result = st.session_state.get("prediction_result")
    selected_model = st.session_state.get("prediction_model_choice", model_choice)
    if result is None:
        st.info("저장된 모델로 예측 실행 버튼을 누르세요.")
        return

    direction_label = "상승" if result["direction"] == "UP" else "하락"
    direction_delta = result["pred_log_return"] * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("사용 모델", f"Model {selected_model}")
    col2.metric("백테스트 방향 정확도", f"{BACKTEST_DA[selected_model]:.2f}%")
    col3.metric("다음 주 예측 방향", direction_label, delta=f"{direction_delta:.2f}%")

    df_plot = result["df_plot"]
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.pyplot(plot_price_history(df_plot, title=f"{asset_name} 종가 추이"))
    with chart_col2:
        st.pyplot(plot_sentiment_index(df_plot["Investor_Sentiment_PC1"], title="투자자 심리지수 PC1"))

    st.subheader("최근 예측 입력값")
    st.dataframe(pd.DataFrame([result["current_data"]]), width='stretch')
