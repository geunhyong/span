# prophet_module.py
import FinanceDataReader as fdr
from prophet import Prophet
import datetime
import pandas as pd

def fetch_stock_data(ticker, years=2):
    """지정된 종목의 n년치 주식 데이터를 불러옵니다."""
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=365 * years)
    
    try:
        df = fdr.DataReader(ticker, start_date, today)
        if df.empty:
            return None, start_date, today
        return df, start_date, today
    except Exception:
        return None, start_date, today

def prepare_prophet_data(df):
    """Prophet 모델에 맞게 데이터프레임 컬럼을 변환합니다."""
    prophet_df = df.reset_index()[['Date', 'Close']]
    prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
    return prophet_df

def run_prophet_forecast(prophet_df, periods=30):
    """Prophet 모델을 학습하고 미래를 예측합니다."""
    model = Prophet(daily_seasonality=False) # type: ignore
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    
    return model, forecast