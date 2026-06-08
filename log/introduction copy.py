import streamlit as st

def render_introduction():
    """프로젝트 소개 페이지"""
    st.title('🎯 Samsung Electronics Sentiment Analysis')
    
    st.markdown("""
    ### 📊 프로젝트 개요
    - **목표**: 기술적 지표와 투자자 심리지수의 관계 분석
    - **대상**: Samsung Electronics (005930.KS)
    - **기간**: 2020-05-25 ~ 2026-05-25 (주간 데이터)
    
    ### 🔬 실험도구
    - **기술 지표**: ATR_10, MFI_10, STOCHk_10_3_3
    - **심리지수**: OLS 잔차 기반 + PCA (설명력 77.81%)
    - **모델**: XGBoost (n_estimators=200, max_depth=5)
    
    ### 🎯 모델 구성
    - **Model A**: Baseline + Investor_Sentiment_PC1 (19 features)
    - **Model B**: Baseline + 3 Scaled Residuals (21 features)
    - **Model C**: Baseline + PC1 + 3 Scaled Residuals (22 features)
    
    ### 📈 주요 특성
    - **PC1 로딩**: [-0.5187, 0.6029, 0.6062]
    - **데이터 포인트**: 107개
    - **검증**: 시계열 교차검증
    """)
    
    st.divider()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Data Points', '107')
    with col2:
        st.metric('PCA Variance', '77.81%')
    with col3:
        st.metric('Models', '3')
