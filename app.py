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
# ==========================================
# 페이지 2: 주식 가격 예측 (Prophet)
# ==========================================
elif menu == "📈 주식 가격 예측":
    st.title("📈 Prophet 활용 주식 종가(Close) 예측")
    
    # ==========================================
    # ✨ [신규] 검색 히스토리 (Session State) 관리
    # ==========================================
    st.sidebar.subheader("🔍 종목 설정")
    
    # 1. 메모리(히스토리 저장소) 초기화: 앱에 처음 접속했을 때 한 번만 실행됨
    if 'ticker_history' not in st.session_state:
        st.session_state.ticker_history = ["005930"] # 기본값: 삼성전자
        
    # 2. 히스토리 선택 박스 (메모리에 저장된 리스트를 보여줌)
    selected_ticker = st.sidebar.selectbox(
        "최근 검색 종목에서 선택", 
        st.session_state.ticker_history
    )
    
    # 3. 새로운 종목 입력칸
    new_ticker = st.sidebar.text_input("새 종목 직접 입력 (예: 015760)", "")
    
    # 4. 최종 실행할 종목 결정 (사용자가 새로 입력한 값이 있으면 그것을 최우선으로 사용!)
    ticker = new_ticker if new_ticker else selected_ticker
    
    run_prophet = st.sidebar.button("주식 예측 실행")
    
    if run_prophet:
        # --- 검색 기록 업데이트 로직 ---
        # 이미 검색했던 종목이면 리스트에서 뺐다가 맨 앞으로 다시 넣음 (최신화)
        if ticker in st.session_state.ticker_history:
            st.session_state.ticker_history.remove(ticker)
        st.session_state.ticker_history.insert(0, ticker)
        
        # 무한정 길어지지 않도록 최근 10개까지만 유지
        st.session_state.ticker_history = st.session_state.ticker_history[:10]
        
        # --------------------------------
        
        with st.spinner(f"종목코드 [{ticker}] 데이터 수집 및 예측 중..."):
            # 1. 모듈에서 데이터 불러오기
            df, start_date, today = pm.fetch_stock_data(ticker, years=5)
            
            # ... (이하 기존 데이터 처리 및 그래프 그리는 로직 그대로 유지) ...            
            if df is None:
                st.error("데이터를 불러오는 데 실패했습니다.")
            else:
                # [신규 추가] 이동평균선 산출 및 시각화
                ma_df = pm.add_moving_averages(df)
                

                # 2. 모듈에서 예측 수행
                prophet_df = pm.prepare_prophet_data(df)
                model, forecast = pm.run_prophet_forecast(prophet_df, periods=30)
                
                # 3. 화면 표출 (클라이언트의 역할)
                st.subheader(f"📊 종목명: {pm.get_stock_name(ticker)} (종목코드: {ticker}) (학습 기간: {start_date} ~ {today})")
                
                st.subheader("미래 30일 예측 그래프")
                fig1 = plot_plotly(model, forecast)


                # 개선 1: Range Slider를 끄면, 사용자가 드래그해서 확대할 때 Y축이 자동으로 맞춰집니다!
                fig1.update_layout(xaxis_rangeslider_visible=False)
                # 개선 2: 메인 차트에서 주말(토, 일)의 빈 공간(Gap)을 완전히 잘라냅니다.
                fig1.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
                fig1.update_layout(xaxis_title="날짜", yaxis_title="종가 (원)", showlegend=False)
                st.plotly_chart(fig1, width='stretch')


                st.subheader("예측 성분 분석 (Components)")
                fig2 = plot_components_plotly(model, forecast)

                st.plotly_chart(fig2, width='stretch')

                st.markdown("##### 📈 최근 6개월 주가 및 이동평균선 (MA)")
                # 차트를 보기 좋게 하기 위해 최근 180일 데이터만 슬라이싱
                # 'Close'와 세 개의 MA 컬럼만 선택하여 라인 차트로 그립니다.
                recent_ma_df = ma_df.tail(180)[['Close', 'MA5', 'MA20', 'MA60']]
                st.line_chart(recent_ma_df)

                golden_crosses = pm.find_golden_cross(ma_df)
                if golden_crosses:
                    st.success(f"✨ 최근 6개월 내 골든 크로스 발생일: **{', '.join(golden_crosses)}**")
                else:
                    st.info("📉 최근 6개월 내 골든 크로스(MA5 상향 돌파)가 발생하지 않았습니다.")