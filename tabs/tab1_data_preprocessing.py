import streamlit as st
import pandas as pd
import FinanceDataReader as fdr

def run():
    st.header("데이터 전처리")
    
    # 삼성전자 데이터 수집
    st.subheader("삼성전자 주가 데이터 수집")
    samsung_data = fdr.DataReader('005930', '2018-01-01')  # 6년간 데이터 수집 (주간 기준)
    st.write(samsung_data.head())
    
    # 결측치 확인
    st.subheader("결측치 확인")
    st.write(samsung_data.isnull().sum())
    
    # 수익률 계산
    samsung_data['Return'] = samsung_data['Close'].pct_change()
    st.write(samsung_data[['Close', 'Return']].head())
    
    # 기초 통계량 분석
    st.subheader("기초 통계량 분석")
    st.write(samsung_data.describe())
























