# app.py
import streamlit as st
import os
from prophet.plot import plot_plotly, plot_components_plotly

# 두 개의 모듈을 모두 불러옵니다.
import qc_module as qm
import prophet_module as pm

# 페이지 기본 설정 (가장 위에 있어야 합니다)
st.set_page_config(page_title="AI Quant 통합 대시보드", layout="wide")

# ==========================================
# 사이드바: 내비게이션 메뉴 구성
# ==========================================
st.sidebar.title("🧭 내비게이션")
menu = st.sidebar.radio("이동할 페이지를 선택하세요:", ["🌡️ 기상 데이터 QC", "📈 주식 가격 예측"])

st.sidebar.divider() # 시각적 구분선

# ==========================================
# 페이지 1: 기상 데이터 QC
# ==========================================
if menu == "🌡️ 기상 데이터 QC":
    st.title("🌡️ 기상 정보 데이터 처리 및 QC 대시보드")
    
    # --- 기존 app.py에 있던 사이드바 설정 ---
    data_path = st.sidebar.text_input("데이터 디렉토리 경로", "./data/seoul_weather/")
    run_button = st.sidebar.button("QC 분석 실행")
    
    if run_button:
        if not os.path.exists(data_path):
            st.error(f"경로를 찾을 수 없습니다: {data_path}")
        else:
            with st.spinner("데이터를 불러오고 QC를 수행 중입니다..."):
                raw_df = qm.load_all_csv(data_path)
                
                if raw_df.empty:
                    st.warning("데이터가 비어있습니다.")
                else:
                    clean_df = qm.apply_qc_logic(raw_df)
                    means = qm.get_aggregated_data(clean_df)
                    
                    # (이곳에 기존에 작성하셨던 st.header, st.line_chart, 탭 구성, 
                    # 출근시간 심층 분석 등의 코드를 그대로 붙여넣으시면 됩니다!)
                    st.success("기상 데이터 QC 완료! (기존 시각화 코드 적용 위치)")

# ==========================================
# 페이지 2: 주식 가격 예측 (Prophet)
# ==========================================
elif menu == "📈 주식 가격 예측":
    st.title("📈 Prophet 활용 주식 종가(Close) 예측")
    st.markdown("과제: **최근 2년**의 데이터를 학습하여 향후 **30일**의 주가를 예측합니다.")
    
    # --- 페이지 2 전용 사이드바 설정 ---
    ticker = st.sidebar.text_input("종목 코드 (예: 삼성전자 005930)", value="005930")
    run_prophet = st.sidebar.button("주식 예측 실행")
    
    if run_prophet:
        with st.spinner("데이터 수집 및 Prophet 예측 중..."):
            # 1. 모듈에서 데이터 불러오기
            df, start_date, today = pm.fetch_stock_data(ticker, years=2)
            
            if df is None:
                st.error("데이터를 불러오는 데 실패했습니다.")
            else:
                # 2. 모듈에서 예측 수행
                prophet_df = pm.prepare_prophet_data(df)
                model, forecast = pm.run_prophet_forecast(prophet_df, periods=30)
                
                # 3. 화면 표출 (클라이언트의 역할)
                st.subheader(f"📊 종목코드: {ticker} (학습 기간: {start_date} ~ {today})")
                
                st.subheader("미래 30일 예측 그래프")
                fig1 = plot_plotly(model, forecast)
                fig1.update_layout(xaxis_title="날짜", yaxis_title="종가 (원)", showlegend=False)
                st.plotly_chart(fig1, use_container_width=True)
                
                st.subheader("예측 성분 분석 (Components)")
                fig2 = plot_components_plotly(model, forecast)
                st.plotly_chart(fig2, use_container_width=True)