# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.data_fetcher import get_recent_data
from modules.predictor import QuantPredictor

# =====================================================================
# 1. 페이지 세팅 및 객체 로드
# =====================================================================
st.set_page_config(page_title="AI 퀀트 다중 모델 대시보드", page_icon="📈", layout="wide")

@st.cache_resource
def load_ai_model():
    return QuantPredictor()

try:
    predictor = load_ai_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"🚨 모델 로드 실패: {e}")

# =====================================================================
# 2. 사이드바 (Control Panel)
# =====================================================================
st.sidebar.title("⚙️ 리서치 제어반")
asset_choice = st.sidebar.selectbox("📊 분석 자산", ["삼성전자", "코스피", "비트코인"])

st.sidebar.divider()
st.sidebar.markdown("**🤖 AI 모델 선택 (A/B/C)**")
model_choice = st.sidebar.radio(
    "적용할 연구 모델을 선택하세요:",
    options=["A", "B", "C"],
    format_func=lambda x: {
        "A": "Model A (PCA 통합 심리)",
        "B": "Model B (세부 잔차 분리) 🏆",
        "C": "Model C (전체 혼용)"
    }[x]
)

analyze_btn = st.sidebar.button("실시간 분석 실행 🚀")

st.title("📈 AI 퀀트: 투자자 심리 기반 다중 모델 대시보드")
st.markdown("회의록 [4장 연구 설계]에 따른 Model A, B, C의 실시간 예측을 비교합니다.")
st.divider()

# =====================================================================
# 3. 메인 로직
# =====================================================================
if analyze_btn and model_loaded:
    with st.spinner(f'{asset_choice} 실시간 데이터 수집 및 {model_choice} 모델 추론 중...'):
        try:
            df_recent = get_recent_data(asset_choice)
            result = predictor.get_prediction(asset_choice, df_recent, model_type=model_choice)
            df_plot = result['df_plot']
            curr_data = result['current_data']
            
            # 모델별 성과 매핑 (train_pipeline.py 테스트 결과 기준)
            da_scores = {"A": 52.94, "B": 56.30, "C": 55.46}
            
            # -------------------------------------------------------------
            # UI: 섹션 1 (투자자 심리 분석)
            # -------------------------------------------------------------
            st.header(f"🔍 [섹션 1] {asset_choice} 투자자 심리 지표 분석")
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("주가 vs 심리지수 흐름 (최근 6개월)")
                fig1 = make_subplots(specs=[[{"secondary_y": True}]])
                fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'], name="종가", line=dict(color="#00BFFF", width=2)), secondary_y=False)
                fig1.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Investor_Sentiment_PC1'], name="통합 심리지수(PC1)", line=dict(color="#FF4500", dash='dot')), secondary_y=True)
                fig1.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig1, width='stretch')

            with col2:
                # 선택된 모델에 따라 다른 차트를 보여줌
                if model_choice == "A":
                    st.subheader("PCA 지표 가중치")
                    labels = ['MFI (추세)', 'Stochastic (단기)', 'NATR (충격)']
                    fig2 = go.Figure(data=[go.Pie(labels=labels, values=[0.688, 0.677, 0.259], hole=.5, marker_colors=['#00D2FF', '#3A7BD5', '#FF416C'])])
                    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", y=-0.2))
                    st.plotly_chart(fig2, width='stretch')
                else:
                    st.subheader("현재 주간(Week) 순수 심리 잔차 (상대 강도)")
                    st.markdown("지표 간 스케일 왜곡을 방지하기 위해, **최근 6개월 내 최댓값 대비 현재 잔차의 비율(-1.0 ~ 1.0)**을 시각화합니다.")
                    x_labels = ['ATR 잔차', 'MFI 잔차', 'STOCH 잔차']
                    
                    # 1. 훈련용 글로벌 스케일러 대신, 시각화 전용 원본 잔차 가져오기
                    raw_vals = [curr_data['ATR_10_res'], curr_data['MFI_10_res'], curr_data['STOCHk_10_3_3_res']]
                    
                    # 2. Local MaxAbs Scaling (최근 6개월 데이터 기준 절댓값의 최댓값 구하기)
                    local_max = [
                        df_plot['ATR_10_res'].abs().max(),
                        df_plot['MFI_10_res'].abs().max(),
                        df_plot['STOCHk_10_3_3_res'].abs().max()
                    ]
                    
                    # 3. 현재 잔차를 최댓값으로 나누어 -1.0 ~ 1.0 사이의 비율(%)로 변환
                    y_vals = [v / m if m != 0 else 0 for v, m in zip(raw_vals, local_max)]
                    
                    colors = ['#FF416C' if v < 0 else '#00D2FF' for v in y_vals]
                    
                    fig2 = go.Figure(data=[go.Bar(
                        x=x_labels, 
                        y=y_vals, 
                        marker_color=colors,
                        text=[f"{v:.2f}" for v in y_vals], # 막대 위에 -1.0 ~ 1.0 비율 표시
                        textposition='auto'
                    )])
                    fig2.update_layout(
                        height=350, 
                        margin=dict(l=0, r=0, t=20, b=0),
                        yaxis=dict(range=[-1.1, 1.1]) # Y축을 -1.1 ~ 1.1로 고정하여 직관성 확보
                    )
                    st.plotly_chart(fig2, width='stretch')

            st.divider()

            # -------------------------------------------------------------
            # UI: 섹션 2 (AI 주가 방향 예측)
            # -------------------------------------------------------------
            st.header(f"🤖 [섹션 2] Model {model_choice} 주가 방향 예측")
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.info(f"💡 Model {model_choice} 백테스트 성능")
                st.metric(label="방향 정확도 (DA)", value=f"{da_scores[model_choice]} %", delta="검증 완료", delta_color="normal")

            with col4:
                direction_txt = "상승 (UP) 🚀" if result['direction'] == "UP" else "하락 (DOWN) 🔻"
                color = "normal" if result['direction'] == "UP" else "inverse"
                
                st.success(f"💡 다음 주 예측 결과")
                st.metric(label="AI 예측 방향", value=direction_txt, 
                          delta=f"예상 로그수익률: {result['pred_log_return']*100:.2f}%", delta_color=color)

            with col5:
                st.subheader("예측 강도 (Signal Strength)")
                strength = min(abs(result['pred_log_return']) * 100 * 20, 100) 
                fig3 = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = strength,
                    title = {'text': "AI 확신도"},
                    gauge = {'axis': {'range': [None, 100]},
                             'bar': {'color': "#FF416C" if result['direction'] == "DOWN" else "#00D2FF"}}
                ))
                fig3.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig3, width='stretch')

        except Exception as e:
            st.error(f"데이터 처리 중 오류가 발생했습니다: {e}")
            st.stop()
else:
    if model_loaded:
        st.info("👈 좌측 제어반에서 분석할 자산과 모델(A/B/C)을 선택하고 실행을 눌러주세요.")
