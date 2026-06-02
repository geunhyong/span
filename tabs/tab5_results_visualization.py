
import streamlit as st
from visualization import plot_actual_vs_predicted

def run():
    st.header("결과 시각화")

    # 예측된 주가 데이터와 실제 데이터 로드
    actual_data = pd.read_csv('actual_data.csv')
    predicted_data = pd.read_csv('predicted_data.csv')

    # 시각화 수행
    plot_actual_vs_predicted(actual_data['Close'], predicted_data['Predicted'])
