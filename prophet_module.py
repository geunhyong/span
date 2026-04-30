# prophet_module.py
import FinanceDataReader as fdr
from prophet import Prophet
import datetime
import pandas as pd

def get_stock_name(ticker):
    """종목 코드를 입력받아 해당 종목의 이름을 반환합니다."""
    try:
        stock_info = fdr.StockListing('KRX')
        stock_name = stock_info[stock_info['Code'] == ticker]['Name'].values[0]
        return stock_name
    except Exception:
        return "알 수 없는 종목"
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
    
    # [수정 1] weekly_seasonality=False 를 명시적으로 추가하여 주간 패턴 학습을 차단합니다.
    model = Prophet(
        daily_seasonality=False, #type:ignore
        weekly_seasonality=False,  # type: ignore
        seasonality_mode='multiplicative', 
        changepoint_prior_scale=0.15
    ) # type: ignore
    
    model.fit(prophet_df)
    
    # [수정 2] 미래 날짜를 만들 때 달력 기준 30일이 아니라, '영업일(Business Day)' 기준 30일로 만듭니다.
    # freq='B' 옵션을 주면 토/일요일을 아예 생성하지 않습니다.
    future = model.make_future_dataframe(periods=periods, freq='B')
    forecast = model.predict(future)
    
    return model, forecast

def add_moving_averages(df):
    """
    원본 주식 데이터프레임에 이동평균선(5일, 20일, 60일) 컬럼을 산출하여 추가합니다.
    """
    ma_df = df.copy()
    
    # Pandas의 rolling() 함수를 사용하여 지정된 기간(window)의 평균을 구합니다.
    ma_df['MA5'] = ma_df['Close'].rolling(window=5).mean()
    ma_df['MA20'] = ma_df['Close'].rolling(window=20).mean()
    ma_df['MA60'] = ma_df['Close'].rolling(window=60).mean()
    
    return ma_df

def find_golden_cross(ma_df, days=180):
    """최근 지정된 기간(days) 내에 발생한 골든 크로스 날짜를 찾습니다."""
    recent_df = ma_df.tail(days).copy()
    
    # 조건: 어제는 MA5 <= MA20 이었고, 오늘은 MA5 > MA20 인 날
    cross_condition = (
        (recent_df['MA5'] > recent_df['MA20']) & 
        (recent_df['MA5'].shift(1) <= recent_df['MA20'].shift(1))
    )
    
    # 조건에 맞는 날짜만 추출하여 리스트로 반환 (예: ['2024-03-15', '2024-04-20'])
    golden_cross_dates = recent_df[cross_condition].index.strftime('%Y-%m-%d').tolist()
    return golden_cross_dates