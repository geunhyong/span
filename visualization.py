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

def plot_stock_data(df):
    """
    조원들이 준 데이터(df)를 받아서
    주가 추이 그래프를 그려 화면에 띄워주는 함수
    """
    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(df['Date'], df['Close'],**ls0, label='주가 추이')
    ax.set_title("주가 추이", **fs0)
    
    ax.set_title("삼성 상장주식 trend", **fs2)
    ax.legend()
    return fig























