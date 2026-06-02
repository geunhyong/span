import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def calculate_indicators(data):
    """
    기술적 지표를 계산하는 함수
    :param data: 주가 데이터프레임
    :return: ATR, MFI, Stochastic
    """
    # ATR 계산
    data['High-Low'] = data['High'] - data['Low']
    data['High-Prev Close'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-Prev Close'] = abs(data['Low'] - data['Close'].shift(1))
    tr = data[['High-Low', 'High-Prev Close', 'Low-Prev Close']].max(axis=1)  # True Range
    data['ATR'] = tr.rolling(window=14).mean()  # 14일 평균 True Range

    # MFI 계산
    positive_flow = (data['Close'].diff() > 0) * data['Volume']  # 매수 압력
    negative_flow = (data['Close'].diff() < 0) * data['Volume']  # 매도 압력
    money_flow_index = 100 * (positive_flow.rolling(window=14).sum() / (positive_flow.rolling(window=14).sum() + negative_flow.rolling(window=14).sum()))  # MFI
    data['MFI'] = money_flow_index

    # Stochastic 계산
    low_min = data['Low'].rolling(window=14).min()
    high_max = data['High'].rolling(window=14).max()
    data['Stochastic'] = 100 * ((data['Close'] - low_min) / (high_max - low_min))  # Stochastic %K

    return data[['ATR', 'MFI', 'Stochastic']]

def run():
    st.header("심리지표 계산")
    
    # 필요한 데이터 로드
    samsung_data = pd.read_csv('samsung_data.csv')  # 예를 들어 미리 저장한 파일을 로드
    indicators_df = calculate_indicators(samsung_data)
    
    st.subheader("계산된 심리지표")
    st.write(indicators_df.head())