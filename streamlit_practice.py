import streamlit as st
import pandas as pd
from prophet.plot import plot_plotly, plot_components_plotly

import prophet_module as pm

st.set_page_config(page_title="Streamlit 연습용 앱", layout="wide")

st.title("📚 Streamlit 연습용 주식 예측 대시보드")
st.markdown(
    "이 앱은 Streamlit 연습용으로 만든 간단한 주식 데이터 조회 및 Prophet 예측 예제입니다."
)

with st.sidebar:
    st.header("설정")
    ticker = st.text_input("종목 코드 입력", value="005930")
    years = st.slider("불러올 과거 데이터 기간(년)", min_value=1, max_value=10, value=2)
    forecast_days = st.slider("예측 기간(영업일)", min_value=7, max_value=90, value=30)
    run_button = st.button("예측 실행")

if run_button:
    if not ticker.strip():
        st.error("종목 코드를 입력해주세요.")
    else:
        with st.spinner(f"종목코드 {ticker} 데이터를 불러오고 예측 중입니다..."):
            df, start_date, today = pm.fetch_stock_data(ticker, years=years)

            if df is None or df.empty:
                st.error("데이터를 불러오지 못했습니다. 종목 코드를 확인하거나 인터넷 연결을 확인해주세요.")
            else:
                st.subheader(f"종목명: {pm.get_stock_name(ticker)}")
                st.write(f"학습 기간: {start_date} ~ {today}")

                st.markdown("### 1. 원본 종가(Close) 데이터")
                st.line_chart(df["Close"])

                st.markdown("### 2. 이동평균선(MA) 추가")
                ma_df = pm.add_moving_averages(df)
                st.line_chart(ma_df[["Close", "MA5", "MA20", "MA60"]].tail(120))

                st.markdown("### 3. Prophet 예측 수행")
                prophet_df = pm.prepare_prophet_data(df)
                model, forecast = pm.run_prophet_forecast(prophet_df, periods=forecast_days)

                plot1 = plot_plotly(model, forecast)
                plot1.update_layout(
                    xaxis_rangeslider_visible=False,
                    title_text="Prophet 예측 결과",
                    xaxis_title="날짜",
                    yaxis_title="종가"
                )
                st.plotly_chart(plot1, use_container_width=True)

                st.markdown("### 4. 예측 성분 분석")
                plot2 = plot_components_plotly(model, forecast)
                st.plotly_chart(plot2, use_container_width=True)

                golden_crosses = pm.find_golden_cross(ma_df)
                if golden_crosses:
                    st.success(f"최근 골든 크로스 발생일: {', '.join(golden_crosses)}")
                else:
                    st.info("최근 골든 크로스가 발견되지 않았습니다.")

else:
    st.info("왼쪽 사이드바에서 종목 코드와 예측 옵션을 입력한 뒤 ‘예측 실행’ 버튼을 눌러주세요.")
