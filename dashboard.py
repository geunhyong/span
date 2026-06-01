import streamlit as st  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns       


st.set_page_config(
    initial_sidebar_state="collapsed",
    layout = "wide",
    page_icon= "📈",
    page_title= "투자자 심리지수 기반 주가 예측 대시보드")



option = st.multiselect("분석할 모델을 선택하세요", ["model_xgb","model_A", "model_B","model_C","naive_baseline","xgb_",""])
st.write(option)




with_file_objects_input = st.file_uploader("파일을 업로드하세요")
if with_file_objects_input:
    st.write("uploaded_file_name :" + with_file_objects_input.name)
    file_contents = with_file_objects_input.read().decode("utf-8")
    st.write(file_contents)





#이 모듈은 Streamlit 대시보드의 화면(UI) 전용 출력을 담당한다.
#데이터 처리, 모델 학습, 예측 계산은 수행하지 않으며,
#이미 준비되어 전송된 context 딕셔너리 내용을 기반으로 위젯만 배치한다.

# ==============================================================================
# 1. 기본 레이아웃 및 환경 설정 영역 (기존 함수 유지/확장)
# ==============================================================================

def render_header(context):
    """
    대시보드 상단 헤더 및 프로젝트 배경 출력

    이 함수는 Streamlit 화면의 가장 위쪽에 표시될 제목과 안내 문구를
    그리는 역할만 담당한다. 데이터 처리, 모델 계산, 예측 생성 같은
    계산 작업은 수행하지 않는다.

    Parameters
    ----------
    context : dict
        프로젝트 전체에서 공유하는 공용 데이터 딕셔너리.
        필요하면 context['config'] 값을 참고하여 제목이나 부제목을 바꿀 수 있다.

    Returns
    -------
    None
        화면에 직접 출력만 수행하므로 반환값은 없다.

    Notes
    -----
    - UI 전용 함수이므로 st.title(), st.write() 같은 출력 함수만 사용한다.
    - 계산 로직은 절대 넣지 않는다.
    """
    title = context.get("config", {}).get(
            "app_title", "투자자 심리지수 기반 주가 예측 대시보드"
    )
    subtitle = context.get("config", {}).get(
        "subtitle", "XGBoost Machine Learning & PCA Multi-Indicator Sentiment Model"
    )

    st.title(title)
    st.caption(subtitle)
    st.markdown("---")




def render_sidebar(context):
    """
    좌측 사이드바: 현재 분석 대상 정보 요약 표기

    이 함수는 사용자가 대시보드 동작에 필요한 옵션을 확인할 수 있도록
    왼쪽 사이드바를 구성한다.

    Parameters
    ----------
    context : dict
        프로젝트 전체에서 공유하는 공용 데이터 딕셔너리.
        사이드바에 표시할 기본값은 context['input'] 또는 context['config']에서 가져올 수 있다.

    Returns
    -------
    None
        화면 출력만 수행하며 값을 반환하지 않는다.

    Notes
    -----
    - UI 요소만 생성해야 한다.
    - 학습, 예측, 전처리 같은 계산은 다른 모듈에서 수행한다.
    """
    with st.sidebar:
        st.header("⚙️ 분석 설정 요약")

        input_data = context.get("input", {})
        ticker = input_data.get("ticker", "삼성전자 (005930)")
        period = input_data.get("period", "최근 6년 (주봉)")
        window = input_data.get("window", "10 (프로젝트 기준 윈도우)")

        st.info(f"**대상 종목:**\n{ticker}")
        st.info(f"**분석 기간:**\n{period}")
        st.info(f"**지표 기준 Window:**\n{window}주")

        st.markdown("---")
        st.markdown("💡 *데이터 수집부터 예측까지 파이프라인 자동화 완료*")


def render_tabs(context):
    """
    최신 기획서 흐름에 맞춘 4대 핵심 탭 구성
    개요(프로젝트 소개) | 데이터 및 전처리(잔차/PCA) | 모델 예측 및 성능 | 시스템 로그
    """
    tabs = st.tabs(
        [
            "📋 프로젝트 개요",
            "📊 데이터 탐색 & 순수 심리(PCA)",
            "🤖 XGBoost 예측 결과",
            "📜 시스템 로그",
        ]
    )
    return tabs

# ==============================================================================
# 2. [탭 1] 프로젝트 소개 영역 (신규 분리)
# ==============================================================================

def render_overview_section(context):
    """
    [신규 생성] 김민호님의 스토리라인 기획 반영.
    - 보조지표(ATR, MFI, STOCH)의 가격 효과 통제(Resampling) 목적 서술
    - 전체 분석 및 검증 프로세스 흐름도/안내문 출력
    """

# ==============================================================================
# 3. [탭 2] 데이터 정제 및 PCA 분석 검증 영역 (김근형님 시각화 집중 반영)
# ==============================================================================
if 'all_data' not in st.session_state:
    with st.spinner("조원들의 엔진(all_might.py)을 가동 중입니다..."):
        all_data = {}
        for symbol, name in all_might.SYMBOLS.items():
            all_data[name] = all_might.process_symbol(symbol, name)
        st.session_state.all_data = all_data

def render_data_prep_section(context):
    """
    조원들이 만든 all_might 데이터를 가져와서 히트맵과 PCA 가중치를 시각화합니다.
    """
    """
    [신규 생성] 가격 요인을 통제한 잔차 분석 및 PCA 결과 출력 전용 함수.
    - 김근형님이 구현할 '잔차 간 Pearson vs Spearman Heatmap' 출력 (`st.pyplot`)
    - 피어슨과 스피어만의 차이점을 설명하는 민호님의 핵심 가이드 텍스트 배치
    - 'PCA 제1주성분(Investor Sentiment) 구성을 위한 지표별 기여 가중치(Weights) 바 차트' 출력
    """
    st.markdown("## 📊 데이터 탐색 & 심리(PCA)")
    #all_data = context.get("data", {}) # 데이터 가방을 엽니다.
    all_data = context["data"]
for name, data in all_data.items():
        df = data[0]
        loadings = data[4]
        
        st.subheader(f"🔍 {name} 분석")
        
        # 1. 히트맵 출력 (all_might의 plot_heatmap 활용)
        # 이 함수는 b64 문자열을 리턴하므로 st.image로 출력합니다.
        b64_hm = all_might.plot_heatmap(name, df)
        if b64_hm:
            st.write("### 잔차 상관계수 히트맵")
            st.image(f"data:image/png;base64,{b64_hm}")
        
        # 2. PCA 가중치 (loadings 데이터를 사용)
        st.write("### PCA 제1주성분 기여 가중치")
        col1, col2, col3 = st.columns(3)
        col1.metric("ATR", f"{loadings[0]:.2f}")
        col2.metric("MFI", f"{loadings[1]:.2f}")
        col3.metric("STOCH", f"{loadings[2]:.2f}")
        
        st.info("💡 **가이드**: ATR, MFI, STOCH 잔차들의 결합이 PC1(투자자 심리지수)을 구성합니다.")

# [섹션 구현: 모델 예측 결과]
def render_metric_section(context):
    st.markdown("## 🤖 모델 예측 성능")
    all_data = context["data"]
    
    for name, data in all_data.items():
        results = data[1] # 성능지표
        st.subheader(f"{name} 성능 지표")
        
        # 모델별 성능표(HTML 방식)를 가져와서 렌더링
        st.components.v1.html(all_might.metrics_table(results), height=300)

        # 2. [추가하면 좋을 부분] 표를 보고 판단하기 힘들 때 친절하게 알려주기
        # Model C의 성공률을 가져와서 판단하는 로직
        model_c = results.get("Model C (전체)")
        if model_c:
            acc = model_c.get("성공률(방향)", 0)
            if acc > 0.53: # 보수적으로 53% 이상이면 아주 훌륭함
                st.success(f"🎉 {name}: 모델 C가 시장 방향성을 효과적으로 포착하고 있습니다! (성공률: {acc:.1%})")
            elif acc > 0.50:
                st.info(f"⚖️ {name}: 시장과 유사한 수준입니다. 심리지수와 결합을 더 최적화해보세요.")
            else:
                st.error(f"⚠️ {name}: 방향성 예측 성능 개선이 필요합니다. (성공률: {acc:.1%})")

def render_figure_section(context):
    all_data = context["data"]
    for name, data in all_data.items():
        pred_data = data[2]
        results = data[1]
        for mname, (pred_s, true_s) in pred_data.items():
            st.write(f"### {mname} 예측 그래프")
            b64_plot = all_might.plot_pred(name, mname, true_s, pred_s, results[mname])
            st.image(f"data:image/png;base64,{b64_plot}")






# ==============================================================================
# 4. [탭 3] XGBoost 모델 예측 및 ML 지표 평가 영역 (권보성님/김민호님 기획 반영)
# ==============================================================================

def render_metric_section(context):
    """
    [기존 수정] 수치 기반의 모델 평가 지표 출력 함수.
    - 기존의 텍스트 무더기(`st.write`) 출력을 탈피
    - 최신 기획서에 명시된 TimeSeriesSplit 기반의 R²(결정계수), RMSE, MAE를 `st.metric` 카드로 구조화
    """

def render_result_section(context):
    """
    [기존 수정] 최종 주가 예측 추세 시각화 출력 함수.
    - XGBoost 모델이 도출한 테스트 데이터의 '실제 주가 vs 예측 주가 시계열 라인 차트' 출력
    - 과적합 방지를 증명하는 '조기 중단(Early Stopping) 학습 곡선(Learning Curve)' 차트 추가 출력 가능 영역
    """


def render_input_section(context):
    """
    사용자가 분석에 필요한 입력값을 확인하거나 수정하는 영역을 출력해 보여준다.
    이 함수는 분석에 사용되는 입력 정보를 보기 좋게 배치한다.
  
    일반적으로 종목명, 날짜 범위, 모델 선택, 실행 버튼 같은
    사용자 입력 요소를 넣는다.

    Parameters
    ----------
    context : dict
        공용 데이터 딕셔너리.
        기본 입력값은 context['input'] 또는 context['config']에서 가져올 수 있다.
        사용자가 선택한 값은 다시 context['input']에 반영할 수 있다.
    
    Returns
    -------
    None
        화면에 직접 위젯을 출력하므로 반환값은 없다.

    Notes
    -----
    - 입력 UI만 담당한다.
    - 입력값 검증이나 데이터 처리 로직은 별도 모듈에서 처리한다.
    """
    
    st.subheader("입력값")
    input_data = context.get("input", {})
    if not input_data:
        st.info("입력값이 아직 없습니다.")
        return

    st.write(f"Ticker: {input_data.get('ticker', 'N/A')}")
    st.write(f"Period: {input_data.get('period', 'N/A')}")



def render_result_section(context):
    """
    모델의 예측 결과나 분석 결과를 화면에 출력한다.

    이 함수는 계산 모듈이 생성한 결과를 사용자에게 보여주는 역할을 한다.
    예측값, 표 형식 결과, 요약 문장 등을 배치할 수 있다.

    Parameters
    ----------
    context : dict
        공용 데이터 딕셔너리.
        결과 데이터는 주로 context['results']에 저장되어 있다고 가정한다.

    Returns
    -------
    None
        결과를 화면에 표시만 하므로 반환값은 없다.

    Notes
    -----
    - 결과를 새로 계산하지 말고, 이미 만들어진 값을 보여주기만 한다.
    - 결과가 없을 경우에는 안내 문구를 출력하는 것이 좋다.
    """
    
    st.subheader("결과")
    results = context.get("results", {})
    if not results:
        st.info("표시할 결과가 없습니다.")
        return

    st.write(results)



def render_metric_section(context):
    """
    모델의 성능 지표나 분석 지표를 화면에 출력한다.
    이 함수는 모델 평가 결과나 분석 지표를 사용자에게 보여주는 역할을 한다.
    예측 정확도, RMSE, R² 같은 수치 지표나, 분석 요약 문장 등을 배치할 수 있다.                         
    Parameters
    ----------                  
    context : dict
        공용 데이터 딕셔너리.
        지표 데이터는 주로 context['metrics']에 저장되어 있다고 가정한다.   
    Returns
    -------
    None
        지표를 화면에 표시만 하므로 반환값은 없다.
    Notes

    - 지표를 새로 계산하지 말고, 이미 만들어진 값을 보여주기만 한다.
    - 지표가 없을 경우에는 안내 문구를 출력하는 것이 좋다.           
    -----
    """    
    #st.write("|기준점|사용|측정대상(심리_proxy)|의미|\n|-----|----|--------|----|\n|multi_navie_baseline|Samsung, xgb_model_A| ATR_10_res, MFI_10_res, STOCHk_10_3_3_res, Investor_Sentiment|얼마나 많이 개선되었나|\n|multi_navie_baseline|Samsung, xgb_model_B|Investor_Sentiment|얼마나 많이 개선되었나|\n|multi_navie_baseline|Samsung, xgb_model_C|ATR_10_res, MFI_10_res, STOCHk_10_3_3_res,Investor_Sentiment|얼마나 많이 개선되었나|\n|baseline_exogenous|Samsung, xgb_model_D|기술지표STOCHk_10_3_3, STOCHd_10_3_3의 골든 크로스|	 기술지표와 심리proxy의 비교|\n" )
    st.subheader("지표")
    metrics = context.get("metrics", {})
    if not metrics:
        st.info("표시할 지표가 없습니다.")
        return
    st.write(metrics)




def render_figure_section(context):
    """
    figures에 저장된 그래프를 Streamlit 화면에 표시한다.

    이 함수는 visualization.py 등에서 생성된 figure 객체를 읽어서,
    figure의 종류에 따라 적절한 Streamlit 출력 함수를 사용한다.

    지원하는 그래프 종류:
    - Plotly figure  -> st.plotly_chart()
    - Matplotlib figure -> st.pyplot()
    Seaborn은 보통 Matplotlib 기반으로 그려지므로 st.pyplot()로 출력한다.
    
    Parameters
    ----------
    context : dict
        공유 context 딕셔너리.
        context["figures"]에는 그래프 객체 또는 그래프 정보가 저장되어 있다고 가정한다.
        예시:
        {
            "price_chart": plotly_fig,
            "volume_chart": mpl_fig,
            "sentiment_chart": seaborn_fig
        }

    Returns
    -------
    None
        화면에 그래프만 출력한다.
    """
  
    import plotly.graph_objects as go
    import matplotlib.figure

    figures = context.get("figures", {})

    if not figures:
        st.info("표시할 그래프가 없습니다.")
        return

    for fig_name, fig_obj in figures.items():
        st.subheader(fig_name)

        # 1) Plotly figure인 경우
        if isinstance(fig_obj, go.Figure):
            st.plotly_chart(fig_obj, use_container_width=True)

        # 2) Matplotlib figure인 경우
        elif isinstance(fig_obj, matplotlib.figure.Figure):
            st.pyplot(fig_obj)

        # 3) Seaborn은 보통 Matplotlib 위에 그려지므로
        #    별도의 figure 객체가 아니라면 현재 활성화된 matplotlib figure를 출력
        else:
            try:
                st.pyplot(fig_obj)
            except Exception:
                st.warning(f"'{fig_name}' 그래프는 현재 지원되는 형식이 아닙니다.")

    return None


# ==============================================================================
# 5. [탭 4] 시스템 인프라 및 자동화 영역 (기존 함수 유지)
# ==============================================================================

def render_log_section(context):
    """
    [기존 유지/이름 명확화] 데이터 수집부터 모델 적재까지 자동화 파이프라인의 백엔드 로그(`context['logs']`) 출력.

    실행 로그를 화면에 출력한다.

    이 함수는 데이터 처리, 모델 실행, 예측 완료 등
    프로젝트 진행 과정에서 생성된 로그를 사용자에게 보여준다.

    Parameters
    ----------
    context : dict
        공용 데이터 딕셔너리.
        로그는 보통 context['logs'] 리스트에 문자열 형태로 저장된다.

    Returns
    -------
    None
        로그를 화면에 출력만 하므로 반환값은 없다.

    Notes
    -----
    - 로그는 디버깅과 검증에 매우 중요하다.
    - 시간 순서대로 출력하면 흐름을 파악하기 쉽다.
    """

    st.subheader("로그")

    logs = context.get("logs", [])
    if not logs:
        st.info("로그가 없습니다.")
        return

    for log in logs:
        st.write(f"- {log}")





import all_might


### 데이터 가져오기
#   조원들의 엔진에서 데이터를 불러오는 함수
def load_all_data():
    all_data = {}
    for symbol, name in all_might.SYMBOLS.items():
        # process_symbol 함수가 조원들의 모든 분석 결과를 리턴합니다.
        data = all_might.process_symbol(symbol, name)
        all_data[name] = data
    return all_data



if 'data_loaded' not in st.session_state:
    with st.spinner("조원들의 엔진(all_might.py)을 가동 중입니다..."):
        # 조원들의 모든 데이터를 한 번에 가져옵니다.
        all_data = {}
        for symbol, name in all_might.SYMBOLS.items():
            all_data[name] = all_might.process_symbol(symbol, name)
        st.session_state.all_data = all_data
        #st.session_state.data_loaded = True

context = {
    "config": {"app_title": "투자자 심리지수 기반 주가 예측"},
    "data": st.session_state.all_data, # 조원들의 엔진 결과물
    "input": {"ticker": "삼성전자", "period": "최근 6년", "window": "10"},
    "logs": ["시스템 엔진 로드 완료", "데이터 랜더링 준비 완료"]
}



render_header(context)
tabs = render_tabs(context)

# 3. 탭별로 섹션 연결
with tabs[0]:
    render_overview_section(context)

with tabs[1]:
    # 조원들이 만든 잔차, PCA 차트가 여기서 출력됩니다.
    render_data_prep_section(context) 
    st.write("본 프로젝트는 주가 데이터에서 심리지수를 추출하여 예측력을 검증합니다.")
with tabs[2]:
    # 모델 성능 지표 및 예측 결과 담기 (조원들의 XGBoost 예측 결과 출력)
    render_metric_section(context)
    render_result_section(context)
    render_figure_section(context) 
    # render_figure_section(context) # 만약 그래프가 figures 딕셔너리에 있다면 사용
with tabs[3]:
    render_log_section(context)