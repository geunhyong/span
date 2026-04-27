import streamlit as st
import qc_module as qm
import os

st.set_page_config(page_title="기상 데이터 QC 시스템", layout="wide")

st.title("🌡️ 기상 정보 데이터 처리 및 QC 대시보드")

# 사이드바 설정
st.sidebar.header("설정")
data_path = st.sidebar.text_input("데이터 디렉토리 경로", "./data/기상정보_서울/")
run_button = st.sidebar.button("데이터 분석 실행")

if run_button:
    if not os.path.exists(data_path):
        st.error(f"경로를 찾을 수 없습니다: {data_path}")
    else:
        with st.spinner("데이터를 불러오고 QC를 수행 중입니다..."):
            # 1. 데이터 로딩
            raw_df = qm.load_all_csv(data_path)
            
            if raw_df.empty:
                st.warning("해당 경로에 CSV 파일이 없거나 데이터가 비어있습니다.")
            else:
                # 2. QC 적용
                clean_df = qm.apply_qc_logic(raw_df)
                
                # 3. 평균 산출
                means = qm.get_aggregated_data(clean_df)

                # --- 결과 출력 ---
                st.header("1. QC 결과 요약")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("원본 데이터")
                    st.line_chart(raw_df['temp'])
                with col2:
                    st.subheader("QC 적용 후 (오류 제거)")
                    st.line_chart(clean_df['temp'])

                st.divider()

                st.header("2. 기간별 평균 기온 (80% 유효성 검사 적용)")
                
                # 탭 구성으로 그래프 표현
                t1, t2, t3, t4 = st.tabs(["1시간 평균", "3시간 평균", "8시간 평균", "일평균"])
                
                with t1:
                    st.line_chart(means['1H'])
                    st.dataframe(means['1H'].dropna().tail())
                with t2:
                    st.line_chart(means['3H'])
                with t3:
                    st.line_chart(means['8H'])
                with t4:
                    st.line_chart(means['1D'])

                st.success("분석이 완료되었습니다.")

                # ==========================================
                # [수정] 아래 코드 전체에 들여쓰기(Tab)를 적용하여
                # if run_button: 과 else: 사이로 집어넣습니다.
                # ==========================================
                st.divider()

                st.header("3. 🏢 출근 시간(06시~09시) 심층 분석")

                # 모듈에서 출근 시간 데이터 가져오기
                weekday_rush_df, weekend_rush_df = qm.analyze_commute_time(clean_df)

                if not weekday_rush_df.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    # 전체 기간에 대한 평균 계산
                    weekday_avg = weekday_rush_df['temp'].mean()
                    weekend_avg = weekend_rush_df['temp'].mean()
                    diff = weekday_avg - weekend_avg
                    
                    with col1:
                        st.metric("평일 출근시간 평균", f"{weekday_avg:.2f} °C")
                    with col2:
                        st.metric("주말 아침(06~09시) 평균", f"{weekend_avg:.2f} °C")
                    with col3:
                        st.metric("평일 - 주말 기온차", f"{diff:.2f} °C", 
                                  delta=f"{diff:.2f} °C", delta_color="inverse")
                        
                    st.subheader("📅 일별 평일 출근시간(06~09시) 평균 기온 추이")
                    # 평일 데이터만 일별로 묶어서 평균 산출 (주말은 NaN이 되므로 dropna 처리)
                    daily_rush_mean = weekday_rush_df['temp'].resample('1D').mean().dropna()
                    st.line_chart(daily_rush_mean)
                else:
                    st.info("선택된 데이터 기간에 평일 06시~09시 데이터가 충분하지 않습니다.")
    
else:
    st.info("왼쪽 사이드바에서 경로를 확인하고 '데이터 분석 실행' 버튼을 클릭하세요.")
