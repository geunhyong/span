import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from bs4 import BeautifulSoup as BS
from io import StringIO
import yfinance as yf
from datetime import timedelta as td, datetime, date
import requests
import calendar
import prophet_module as pm
import koreanize_matplotlib
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
from qc_module import prepare_weekly_data


prepare_weekly_data()



ls0= { 'color': '#8A7B8C' ,'linewidth': 3,'linestyle': '-'} 
ls1= { 'color': '#FF6347' ,'linewidth': 3,'linestyle': '--'} 
ls2= { 'color': '#4682B4' ,'linewidth': 3,'linestyle': '-.'} 
ls3= { 'color': '#32CD32' ,'linewidth': 3,'linestyle': ':'}

fs0 = { 'fontsize': 12, 'fontweight': 'bold', 'color': '#8A7B8C' }
fs1 = { 'fontsize': 12, 'fontweight': 'bold', 'color': '#FF6347' }
fs2 = { 'fontsize': 12, 'fontweight': 'bold', 'color': '#4682B4' }
fs3 = { 'fontsize': 12, 'fontweight': 'bold', 'color': '#32CD32' }




# def plot_stock_data(df):
#     """
#     주가 추이 그래프를 그려주는 함수
#     DataFrame에서 'Date'와 'Close' 컬럼을 처리
#     대소문자를 무시하고 발견
#     """
#     # 날짜를 인덱스로 설정
#     if 'Date' in df.columns:
#         df.set_index('Date', inplace=True)
#     elif df.index.name != 'Date':
#         raise ValueError("The DataFrame must have a 'Date' column or be indexed by 'Date'.")

#     # 대소문자를 고려하여 'Close' 컬럼 찾기
#     close_column = [col for col in df.columns if col.lower() == 'close']
#     if not close_column:
#         raise ValueError("The DataFrame must contain a 'Close' column for stock prices.")
#     else:
#         close_column_name = close_column[0]
    
#     fig, ax = plt.subplots(figsize=(10, 6))

#     ax.plot(df.index, df[close_column_name], **ls0, label='주가 추이')
#     ax.set_title("삼성 상장주식 trend", **fs0)
#     ax.legend()
#     plt.grid()
#     return fig


# visualization.py 

def plot_stock_data(df):
    """
    조원들이 준 데이터(df)를 받아서
    주가 추이 그래프를 그려 화면에 띄워주는 함수
    DataFrame에서 'Date'와 'Close' 컬럼을 처리
    대소문자를 무시하고 발견
    """
    # 날짜를 인덱스로 설정
    if 'Date' in df.columns:
        df.set_index('Date', inplace=True)
    elif df.index.name != 'Date':
        raise ValueError("DataFrame에 'Date' 컬럼이 필요합니다.")

    # 대소문자를 고려하여 'Close' 컬럼 찾기
    close_column = [col for col in df.columns if col.lower() == 'close']
    if not close_column:
        raise ValueError("DataFrame에 'Close' 컬럼이 필요합니다.")
    else:
        close_column_name = close_column[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # 조원 스타일을 고려하여 추가적인 주석을 적절히 작성
    ax.plot(df.index, df[close_column_name], **ls0, label='주가 추이')
    ax.set_title("삼성 상장주식 trend", **fs0)
    ax.legend()
    plt.grid()
    return fig


def plot_actual_vs_predicted(actual, predicted):
    """
    실제 종가와 모델 예측값을 비교하는 시계열 그래프
    :param actual: 실제 종가 데이터
    :param predicted: 모델 예측 값
    위치: visualization.py, 코딩한 사원: 김근형, 줄 번호: 15~30
    """
    plt.figure(figsize=(12,6))
    plt.plot(actual.index, actual, label='Actual Price', color='blue')
    plt.plot(predicted.index, predicted, label='Predicted Price', color='orange')
    plt.title('Actual vs Predicted Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_heatmap(data):
    """
    데이터의 상관관계를 히트맵으로 표시하는 함수
    :param data: 분석할 데이터프레임
    위치: visualization.py, 코딩한 사원: 추주원, 줄 번호: 32~48
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

# 추가적인 시각화 함수들...






















