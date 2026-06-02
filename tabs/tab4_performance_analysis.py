import streamlit as st
import pandas as pd

def run():
    st.header("성능 분석")

    # 데이터 로드
    data = pd.read_csv('performance_results.csv')  # 저장된 성능 결과 로드
    st.subheader("성능 지표")
    st.write(data)

    # 히트맵으로 성과 시각화
    import visualization
    visualization.plot_heatmap(data.corr())
