from pathlib import Path

import streamlit as st


def render_process_flow() -> None:
    st.header("프로세스 흐름")

    st.markdown(
        """
1. 삼성전자·KOSPI·Bitcoin의 주봉 가격 데이터 수집
2. 각 자산의 현재 종가와 과거 1~5주 로그수익률 생성
3. 삼성전자 ATR·MFI·Stochastic 기술지표 계산
4. RET·MOM·VOL로 설명되는 부분을 통제하여 residual 3개 생성
5. residual 3개를 표준화하고 PCA로 PC1 통합 심리 proxy 생성
6. **실험 1**: Price-only와 Model B/C/D 비교
7. **실험 2**: 동일한 가격 입력 18개에 residual을 하나씩 추가한 Model A-1/A-2/A-3 비교
8. **실험 3**: Bitcoin을 제외한 KOSPI-only 12개 입력과 Bitcoin을 포함한 Price-only 18개 입력 비교
9. 동일한 58주 테스트 구간과 5개 세부 기간에서 방향 정확도 평가
10. 저장 모델 B/C/D로 최근 1회 예측하고 실시간 계산값과 공식 CSV 값을 함께 표출
        """
    )


def log_process_flow(
    path: str = "process_flow.md",
) -> None:
    content = (
        "# 프로세스 흐름\n\n"
        "1. 주봉 데이터 수집\n"
        "2. 현재 종가 및 1~5주 로그수익률 생성\n"
        "3. 기술지표와 residual 생성\n"
        "4. PCA 기반 PC1 생성\n"
        "5. 실험 1: Price-only와 B/C/D 비교\n"
        "6. 실험 2: A-1/A-2/A-3 비교\n"
        "7. 실험 3: KOSPI-only와 KOSPI+Bitcoin 비교\n"
        "8. 전체 및 기간별 방향 정확도 평가\n"
        "9. 저장 모델 최근 예측\n"
    )

    Path(path).write_text(
        content,
        encoding="utf-8",
    )
