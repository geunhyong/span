# modules/data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =====================================================================
# 자산 매핑 딕셔너리
# =====================================================================
TICKER_MAP = {
    "삼성전자": "005930.KS",
    "코스피": "^KS11",
    "비트코인": "BTC-USD"
}

# =====================================================================
# 보조지표 계산 함수 (훈련 데이터와 100% 동일한 로직 적용)
# =====================================================================
def calc_mfi(df, period=14):
    """ Money Flow Index (MFI) 계산 """
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    rmf = tp * df['Volume']
    diff = tp.diff()
    
    pos_mf = np.where(diff > 0, rmf, 0)
    neg_mf = np.where(diff < 0, rmf, 0)
    
    pos_mf_sum = pd.Series(pos_mf, index=df.index).rolling(window=period).sum()
    neg_mf_sum = pd.Series(neg_mf, index=df.index).rolling(window=period).sum()
    
    with np.errstate(divide='ignore', invalid='ignore'): # 0으로 나누기 방지
        mfr = pos_mf_sum / neg_mf_sum
        mfi = np.where(neg_mf_sum == 0, 100, 100 - (100 / (1 + mfr)))
        
    return pd.Series(mfi, index=df.index)

def calc_atr(df, period=14):
    """ 일반 ATR 계산 (조정 NATR은 predictor.py에서 수행) """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Wilder's Smoothing 방식
    atr = np.zeros_like(tr)
    atr[period-1] = tr[:period].mean() 
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period - 1) + tr.iloc[i]) / period
        
    atr_series = pd.Series(atr, index=df.index)
    atr_series[:period-1] = np.nan
    return atr_series

def calc_stoch(df, n=14, m=3):
    """ Slow Stochastic %K 계산 """
    low_min = df['Low'].rolling(window=n).min()
    high_max = df['High'].rolling(window=n).max()
    
    fast_k = 100 * (df['Close'] - low_min) / (high_max - low_min)
    slow_k = fast_k.rolling(window=m).mean() # Smoothing (3)
    return slow_k

# =====================================================================
# 클라이언트(app.py) 호출용 메인 API
# =====================================================================
def get_recent_data(asset_name):
    """
    선택된 자산의 최근 데이터를 수집하여 지표를 계산한 후 반환합니다.
    """
    ticker = TICKER_MAP.get(asset_name)
    if not ticker:
        raise ValueError("지원하지 않는 자산입니다.")
        
    # 예측 시차(5) + 지표계산(10) + 모멘텀/분산(11)을 여유있게 커버하기 위해 240일 수집
    start_date = datetime.now() - timedelta(days=240)
    
    df = yf.download(ticker, start=start_date, interval='1wk', progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df = df.dropna(subset=['Close'])
    
    if 'Volume' not in df.columns or df['Volume'].isnull().all():
        df['Volume'] = 1 
        
    # 💡 [핵심 수정] 회의록 규약에 맞춰 14 -> 10으로 기간 파라미터 및 컬럼명 일괄 변경
    df['MFI_10'] = calc_mfi(df, 10)
    df['ATR_10'] = calc_atr(df, 10)
    df['STOCHk_10_3_3'] = calc_stoch(df, 10, 3)
    
    # 지표 계산을 위해 사용된 앞부분의 NaN 행 제거
    df_clean = df.dropna(subset=['Close', 'MFI_10', 'ATR_10', 'STOCHk_10_3_3']).copy()
    
    return df_clean