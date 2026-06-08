import streamlit as st


def render_process_flow() -> None:
    st.header("프로세스 흐름")
    st.markdown(
        """
        1. 삼성전자 · 코스피 · 비트코인 주봉 가격 데이터 수집
        2. MFI, ATR, Stochastic 지표 계산
        3. 수익률, 모멘텀, 변동성으로 OLS 잔차 추출
        4. 잔차 3개를 Scaling 후 PCA로 통합 심리지수 생성
        5. Price-only Baseline 및 XGBoost Model A/B/C 비교
        6. 각 자산별 다음 주 로그수익률의 오차와 방향 정확도 평가
        """
    )


def log_process_flow(path: str = "process_flow.md") -> None:
    content = (
        "# 프로세스 흐름\n\n"
        "1. 데이터 수집\n"
        "2. 데이터 전처리\n"
        "3. 심리지표 계산\n"
        "4. 모델 구축 및 검증\n"
        "5. 성능 평가 및 시각화\n"
    )
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)
