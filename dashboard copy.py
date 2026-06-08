import streamlit as st

from log.introduction import render_introduction
from log.process_flow import render_process_flow
from tabs import data_preprocessing, modeling_validation, sentiment_proxy


# =========================================================
# 프로젝트 공통 설정
# =========================================================
APP_TITLE = "투자자 심리지수 기반 주가 예측 대시보드"
PREDICTION_TARGET = "삼성전자"

DATA_ASSETS = [
    "삼성전자",
    "코스피",
    "비트코인",
]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 공통 세션 상태
# =========================================================
def initialize_session_state() -> None:
    """
    대시보드 전체에서 공유하는 기본 session_state를 설정한다.

    asset_name은 데이터 전처리와 심리지표 화면에서
    불러오고 확인할 주봉 데이터의 자산명이다.

    최종 예측 대상은 삼성전자로 고정한다.
    """
    if "asset_name" not in st.session_state:
        st.session_state.asset_name = PREDICTION_TARGET

    st.session_state.prediction_target = PREDICTION_TARGET

    if "raw_data_by_asset" not in st.session_state:
        st.session_state.raw_data_by_asset = {}

    if "sentiment_data_by_asset" not in st.session_state:
        st.session_state.sentiment_data_by_asset = {}

    if "explained_variance_by_asset" not in st.session_state:
        st.session_state.explained_variance_by_asset = {}


# =========================================================
# 사이드바
# =========================================================
def render_default_sidebar() -> None:
    """
    프로젝트 소개·데이터 전처리·심리지표 탭에서 사용하는
    기존 자산 선택 사이드바입니다.
    """
    st.sidebar.header("주봉 데이터 준비")

    asset_name = st.sidebar.selectbox(
        "불러올 주봉 데이터",
        options=[
            "삼성전자",
            "코스피",
            "비트코인",
        ],
        key="sidebar_asset_name",
    )

    st.session_state.asset_name = asset_name

    st.sidebar.caption(
        "선택한 자산의 주봉 데이터와 기술지표 계산 결과를 확인합니다."
    )


def render_modeling_sidebar() -> None:
    """
    모델링/검증 탭에서만 사용하는 고정 안내 사이드바입니다.
    """
    st.session_state.asset_name = "삼성전자"

    st.sidebar.header("분석 대상")

    st.sidebar.markdown("**최종 예측 대상**")
    st.sidebar.markdown("### 삼성전자")

    st.sidebar.markdown("**시장 보조 입력**")
    st.sidebar.write("KOSPI · Bitcoin")

    st.sidebar.caption(
        "삼성전자의 다음 주 로그수익률 방향을 예측하며, "
        "KOSPI와 Bitcoin은 시장 흐름을 반영하는 "
        "보조 입력자료로 사용합니다."
    )

# =========================================================
# 대시보드 상단
# =========================================================
def render_dashboard_header() -> None:
    """
    프로젝트의 목적을 간결하게 소개한다.
    """
    st.title(APP_TITLE)

    st.caption(
        "시장 데이터와 심리 proxy를 활용한 "
        "삼성전자 다음 주 로그수익률 방향 예측"
    )

    st.info(
        "본 프로젝트는 삼성전자를 최종 예측 대상으로 설정합니다. "
        "KOSPI와 Bitcoin의 주봉 데이터는 삼성전자 방향 예측을 위한 "
        "시장학습 보조자료로 사용합니다."
    )


# =========================================================
# 프로젝트 소개 탭의 실험 구성
# =========================================================
def render_experiment_summary() -> None:
    """
    프로젝트 소개 탭에서 모델의 연구 질문 3가지에 대한 간략 설명.

    자세한 성능 결과와 해석은 모델링/검증 탭에서 표시한다.
    """
    st.subheader("실험 구성")

    st.markdown(
        """
        **공통 기본 학습데이터**  
        삼성전자·KOSPI·Bitcoin의 현재 주봉 종가와 과거 1~5주 수익률
        """
    )

    st.markdown(
        """
        **0번 · Price-only**  
        공통 기본 학습데이터만 사용한 가격 기반 비교 모델
        """
    )

    st.markdown(
        """
        **1번 · Model A**  
        공통 기본 학습데이터에 PC1 투자심리 지수를 추가
        """
    )

    st.markdown(
        """
        **2번 · Model B**  
        공통 기본 학습데이터에 ATR·MFI·Stochastic Residual 3개를 추가
        """
    )

    st.markdown(
        """
        **3번 · Model C**  
        공통 기본 학습데이터에 PC1과 Residual 3개를 함께 추가
        """
    )

    st.caption(
        "모든 모델은 삼성전자 다음 주 로그수익률의 상승·하락 방향을 예측하며, "
        "Directional Accuracy를 중심으로 비교합니다."
    )


# =========================================================
# 메인 화면
# =========================================================
def main() -> None:
    initialize_session_state()
    render_dashboard_header()

    overview_tab, data_tab, sentiment_tab, modeling_tab = st.tabs(
        [
            "프로젝트 소개",
            "데이터 전처리",
            "심리지표",
            "모델링/검증",
        ],
        key="dashboard_tabs",
        on_change="rerun",
    )

    # -----------------------------------------------------
    # 프로젝트 소개
    # -----------------------------------------------------
    if modeling_tab.open:
        render_modeling_sidebar()
    else:
        render_default_sidebar()
    with overview_tab:
        render_introduction()

        st.markdown(
            "<div style='height: 24px;'></div>",
            unsafe_allow_html=True,
        )

        render_process_flow()

        st.markdown(
            "<div style='height: 32px;'></div>",
            unsafe_allow_html=True,
        )

        render_experiment_summary()

    # -----------------------------------------------------
    # 데이터 전처리
    # -----------------------------------------------------
    with data_tab:
        data_preprocessing.run()

    # -----------------------------------------------------
    # 심리지표
    # -----------------------------------------------------
    with sentiment_tab:
        sentiment_proxy.run()

    # -----------------------------------------------------
    # 모델링 / 검증
    # -----------------------------------------------------
    with modeling_tab:
        modeling_validation.run()


if __name__ == "__main__":
    main()