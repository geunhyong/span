import streamlit as st


def render_introduction() -> None:
    st.header("프로젝트 소개")
    st.write(
        "이 프로젝트는 가격 데이터에서 기술지표를 계산하고, 가격 요인으로 설명되지 않는 "
        "잔차를 투자자 심리의 대리변수로 사용해 다음 주 수익률 방향을 예측합니다."
    )


def write_introduction(path: str = "introduction.md") -> None:
    content = (
        "# 프로젝트 소개\n\n"
        "가격 데이터, 심리지표, PCA, XGBoost를 이용해 주가 방향 예측 과정을 정리합니다.\n"
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
