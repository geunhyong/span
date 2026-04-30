import streamlit as st
import FinanceDataReader as fdr
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import datetime
import pandas as pd

# 페이지 기본 설정
st.set_page_config(page_title="주식 가격 예측 대시보드", layout="wide")

st.title("📈 Prophet 활용 주식 종가(Close) 예측")
st.markdown("과제: **최근 2년**의 데이터를 학습하여 향후 **30일**의 주가를 예측합니다.")

# 사이드바 사용자 입력
st.sidebar.header("설정")
ticker = st.sidebar.text_input("종목 코드 (예: 삼성전자 005930, 카카오 035720)", value="005930")

if st.sidebar.button("데이터 분석 및 예측 실행"):
    with st.spinner("데이터를 불러오고 예측을 수행 중입니다..."):
        
        # 1. 날짜 설정 (오늘 기준으로 2년 전 데이터부터 가져오기)
        today = datetime.date.today()
        start_date = today - datetime.timedelta(days=365 * 2) # 정확히 2년(730일)
        
        # 2. 데이터 불러오기 (FinanceDataReader 사용)
        try:
            df = fdr.DataReader(ticker, start_date, today)
        except Exception as e:
            st.error("데이터를 불러오는 데 실패했습니다. 종목 코드를 확인해 주세요.")
            st.stop()
            
        if df.empty:
            st.warning("해당 기간의 데이터가 없습니다.")
            st.stop()

        # 3. Prophet 모델을 위한 데이터 전처리 (ds와 y 컬럼 필수)
        # 종가(Close) 컬럼만 사용
        prophet_df = df.reset_index()[['Date', 'Close']]
        prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)
        
        # 4. Prophet 모델 학습
        # 주식 데이터는 주말이 없으므로 일간 계절성은 끄고 학습합니다.
        # model = Prophet(daily_seasonality=False)
        model = Prophet() # 기본값인 auto로 일간 계절성 여부를 결정하도록 할 수도 있습니다.
        model.fit(prophet_df)
        
        # 5. 미래 30일 데이터 예측
        future = model.make_future_dataframe(periods=30)
        # 주말(토, 일)을 제외하려면 아래 코드를 추가할 수 있으나, 기본적으로 Prophet이 추세를 이어갑니다.
        forecast = model.predict(future)
        
        # --- Streamlit 표출 ---
        st.subheader(f"📊 종목코드: {ticker} (기간: {start_date} ~ {today})")
        
        # 원본 데이터 표시
        with st.expander("원본 데이터 확인 (최근 5일)"):
            st.dataframe(df.tail())
            
        # 예측 결과 시각화 (Plotly 사용으로 인터랙티브한 그래프 생성)
        st.subheader("미래 30일 예측 그래프")
        fig1 = plot_plotly(model, forecast)
        fig1.update_layout(xaxis_title="날짜", yaxis_title="종가 (원)", showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
        # 예측 성분 (트렌드, 주간/연간 계절성) 시각화
        st.subheader("예측 성분 분석 (Components)")
        fig2 = plot_components_plotly(model, forecast)
        st.plotly_chart(fig2, use_container_width=True)
        
        # 예측 데이터 표출
        st.subheader("예측 데이터 상세 (향후 30일)")
        forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
        forecast_display.rename(columns={
            'ds': '날짜', 
            'yhat': '예측 종가', 
            'yhat_lower': '예측 최저가', 
            'yhat_upper': '예측 최고가'
        }, inplace=True)
        st.dataframe(forecast_display)
        
        st.success("예측이 성공적으로 완료되었습니다! 그래프를 확대/축소하여 확인해 보세요.")