import streamlit as st


def render_introduction() -> None:
    """프로젝트 소개 화면을 렌더링한다."""
    st.title("🎯 Samsung Electronics Sentiment Analysis")

    st.markdown(
        """
### 📊 프로젝트 개요
- **목표**: 시장 가격정보와 투자자 심리 proxy가 삼성전자 다음 주 방향 예측에 제공하는 추가 정보를 검증
- **최종 예측 대상**: 삼성전자
- **외생 보조 입력**: KOSPI, Bitcoin
- **분석 주기**: 주봉
- **공식 결과 기준**: 프로젝트 루트의 최신 백테스트 CSV

### 🔬 분석 도구
- **가격 feature**: 현재 주봉 종가와 과거 1~5주 로그수익률
- **기술지표**: ATR_10, MFI_10, STOCHk_10_3_3
- **심리 proxy**: RET·MOM·VOL 통제 후 생성한 residual
- **통합 심리지수**: residual 3개 표준화 후 PCA의 PC1
- **예측모델**: XGBoost
- **평가 지표**: R², RMSE, MAE, Directional Accuracy

### 🧪 최종 실험 구성
- **실험 1**: Price-only와 Model B/C/D의 심리 proxy 구성 비교
- **실험 2**: Model A-1/A-2/A-3의 개별 residual 기여도 비교
- **실험 3**: KOSPI-only 12개 입력과 KOSPI+Bitcoin 18개 입력 비교

### 🎯 최종 모델 명칭
- **Model A-1**: ATR residual
- **Model A-2**: MFI residual
- **Model A-3**: Stochastic residual
- **Model B**: PC1
- **Model C**: residual 3개
- **Model D**: PC1 + residual 3개
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sentiment Rows",
        "107",
    )

    col2.metric(
        "Test Weeks",
        "58",
    )

    col3.metric(
        "Compared Rows",
        "10",
    )
