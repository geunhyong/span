import streamlit as st

from log.introduction import render_introduction
from log.process_flow import render_process_flow
from tabs import data_preprocessing, modeling_validation, sentiment_proxy


st.set_page_config(
    page_title="Investor Sentiment Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def render_sidebar() -> None:
    st.sidebar.header("분석 설정")
    st.sidebar.selectbox(
        "분석 대상",
        ["삼성전자", "코스피", "비트코인"],
        key="asset_name",
    )
    st.sidebar.caption("각 탭은 위 설정과 세션 데이터를 공유합니다.")


def main() -> None:
    render_sidebar()

    st.title("투자자 심리지수 기반 주가 예측 대시보드")
    st.caption("데이터 전처리 → 심리지표 계산 → 모델 구축 및 검증")

    overview_tab, data_tab, sentiment_tab, modeling_tab = st.tabs(
        ["프로젝트 소개", "데이터 전처리", "심리지표", "모델링/검증"]
    )

    with overview_tab:
        render_introduction()
        render_process_flow()

    with data_tab:
        data_preprocessing.run()

    with sentiment_tab:
        sentiment_proxy.run()

    with modeling_tab:
        modeling_validation.run()


if __name__ == "__main__":
    main()
