import streamlit as st
import all_might

APP_TITLE = "투자자 심리지수 기반 주가 예측"
APP_SUBTITLE = "다음 주 수익률 및 방향성 중심"

MODEL_SOURCE_NOTE = (
    "현재 화면은 발표 시각화 참고용 대시보드 구조이며, "
    "최종 모델링 결과는 train_pipeline.py 및 저장된 pkl 모델 기준으로 연결 예정입니다."
)

st.set_page_config(
    initial_sidebar_state="collapsed",
    layout = "wide",
    page_icon= "📈",
    page_title= APP_TITLE)



#option = st.multiselect("분석할 모델을 선택하세요", ["model_xgb","model_A", "model_B","model_C","naive_baseline","xgb_",""])
#st.write(option)




#with_file_objects_input = st.file_uploader("파일을 업로드하세요")
#if with_file_objects_input:
#    st.write("uploaded_file_name :" + with_file_objects_input.name)
#    file_contents = with_file_objects_input.read().decode("utf-8")
#    st.write(file_contents)





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
            "app_title", APP_TITLE
    )
    subtitle = context.get("config", {}).get(
        "subtitle", APP_SUBTITLE
    )

    st.title(title)
    st.caption(subtitle)

    st.info(
        context.get("config", {}).get(
            "model_source_note",
            MODEL_SOURCE_NOTE
        )
    )

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
        ticker = input_data.get("ticker", "비트코인 · 삼성전자 · 코스피")
        period = input_data.get("period", "최근 6년 (주봉)")
        window = input_data.get("window", "10 (프로젝트 기준 윈도우)")

        st.info(f"**대상 종목:**\n{ticker}")
        st.info(f"**분석 기간:**\n{period}")
        st.info(f"**지표 기준 Window:**\n{window}")

        st.markdown("---")
        st.markdown("💡 *데이터 수집·전처리·모델링 흐름을 발표용으로 정리 중입니다.*")

        st.markdown("---")
        st.subheader("🧩 최종 실험 기준")

        config = context.get("config", {})

        st.info(f"""
        **전처리 기준**  
        {config.get("final_preprocessing", "확인 중")}

        **모델링 기준**  
        {config.get("final_modeling", "확인 중")}

        **데이터 구조**  
        {config.get("data_structure", "확인 중")}

        **예측 타깃**  
        {config.get("target", "확인 중")}

        **평가 기준**  
        {config.get("evaluation", "확인 중")}
        """)

def render_tabs(context):
    """
    최신 기획서 흐름에 맞춘 4대 핵심 탭 구성
    개요(프로젝트 소개) | 데이터 및 전처리(잔차/PCA) | 모델 예측 및 성능 | 시스템 로그
    """
    tabs = st.tabs(
        [
            "📋 프로젝트 개요",
            "📊 데이터 탐색 & 심리_proxy(PCA)",
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
    프로젝트 개요 및 연구 흐름 설명
    발표용/보고서용 스토리라인 기반 구성
    """

    st.header("📋 프로젝트 개요")
    st.info("""
    본 프로젝트는 **비트코인, 삼성전자, 코스피** 데이터를 함께 활용한  
    **패널 데이터 구조**를 기반으로 합니다.

    `Samsung_crawling.py`를 통해 주봉 OHLCV 데이터와  
    ATR, MFI, Stochastic 지표를 구성하고,  
    `train_pipeline.py`에서 residual 추출, PCA 기반 투자자 심리지수 생성,  
    XGBoost Model A/B/C 학습 및 pkl 모델 저장을 수행하는 구조입니다.

    예측 대상은 절대 종가가 아니라 **다음 주 주봉 로그 수익률**이며,  
    방향성 성공률과 오차 지표를 함께 확인하는 것을 목표로 합니다.
    """)

    st.markdown("### 프로젝트 한눈에 보기")
    st.markdown("### 🤖 최종 모델 구조")
    st.info("""
    **최종 모델 구조 요약**

    - **Model A**: Log Return lag 1~5 + Asset 더미 + Investor_Sentiment_PC1  
    - **Model B**: Log Return lag 1~5 + Asset 더미 + ATR/MFI/STOCH residual 3개  
    - **Model C**: Log Return lag 1~5 + Asset 더미 + Investor_Sentiment_PC1 + residual 3개  

    ※ pkl 파일 확인 결과, Model A/B/C는 각각 XGBRegressor 객체로 저장되어 있으며  
    입력 feature 수는 A=9개, B=11개, C=12개로 확인되었습니다.
    """)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="분석 데이터",
            value="BTC · 삼성전자 · 코스피"
        )

    with col2:
        st.metric(
            label="심리 구성",
            value="Residual + PCA"
        )

    with col3:
        st.metric(
            label="예측 모델",
            value="XGBoost A/B/C"
        )

    with col4:
        st.metric(
            label="예측 타깃",
            value="t+1 로그 수익률"
        )

    st.markdown("---")
    st.markdown("### 🔍 본 프로젝트의 핵심 질문")

    q1, q2, q3 = st.columns(3)

    with q1:
        st.info("""
        **Q1. 왜 기술지표를 그대로 쓰지 않았을까?**

        ATR·MFI·STOCH는 이미  
        가격 변화, 변동성, 거래량 등  
        시장 일반 흐름의 영향을 포함하고 있기 때문
        """)

    with q2:
        st.info("""
        **Q2. 왜 Residual과 PCA를 사용했을까?**

        일반 가격 요인으로 충분히 설명되지 않는 부분을  
        심리 proxy에 가까운 성분으로 탐색하고,  
        공통 흐름을 PC1으로 요약하기 위해
        """)

    with q3:
        st.info("""
        **Q3. 왜 Model A/B/C를 비교했을까?**

        통합 심리지수(PC1), 개별 residual,  
        그리고 두 조합의 설명력을 비교하기 위해
        """)

    st.markdown("---")
    # -------------------------------------------------
    # 1. 프로젝트 목적
    # -------------------------------------------------
    with st.expander("🎯 프로젝트 목적", expanded=True):



        st.markdown("""
        기존 주가 예측은 가격 변화 자체나 재무 정보 중심 접근이 많았습니다.  
        반면 본 프로젝트는 시장 참여자의 심리적 반응과 가까운 흐름이  
        가격 움직임에 일부 영향을 줄 수 있다는 가능성을 바탕으로 시작되었습니다.

        이에 따라 ATR, MFI, Stochastic과 같은 기술지표에서  
        시장 일반 흐름의 영향을 일부 통제하고, 남는 신호를 심리적 반응과
        가까운 proxy로 해석해볼 수 있는지 탐색하고자 하였습니다.
        """)

    # -------------------------------------------------
    # 2. 이론적 배경
    # -------------------------------------------------
    with st.expander("📚 이론적 배경 및 지표 선정 이유"):

        st.markdown("""
        본 프로젝트는 기술지표를 투자심리를 직접 측정하는 지표로 보기보다는,  
        시장 참여자의 행동 흔적을 간접적으로 반영할 가능성이 있는 신호로 해석하였습니다.

        - **ATR (Average True Range)**  
        → 시장 변동성 반응과 가까운 성격을 가질 수 있으며,  
        투자자의 불안·과민 반응이 일부 반영될 가능성을 추정할 수 있습니다.

        - **MFI (Money Flow Index)**  
        → 거래량과 가격 흐름을 함께 고려하므로,  
        자금 유입 및 매수·매도 심리 변화와 가까운 특징을 가질 수 있습니다.

        - **Stochastic Oscillator**  
        → 과매수·과매도 구간에서 나타나는 심리 반응을  
        간접적으로 시사하는 신호에 가까울 수 있다고 보았습니다.
        """)

    # -------------------------------------------------
    # 3. Residual
    # -------------------------------------------------
    with st.expander("🧩 Residual(잔차)를 활용한 이유"):

        st.markdown("""
        기술지표 자체는 이미 가격 변화, 변동성, 거래량 등의  
        시장 일반 흐름 영향을 포함하고 있습니다.

        따라서 기술지표를 그대로 사용하는 경우,  
        순수한 심리 반응만을 설명한다고 보기에는 어려움이 존재할 수 있습니다.

        이에 본 프로젝트에서는 시장 흐름으로 설명되는 부분을 일부 통제하고,  
        설명되지 않는 잔차(residual)를 투자 심리와 가까운 성분으로
        추정해볼 수 있다는 방향성을 고려하였습니다.
        """)

    # -------------------------------------------------
    # 4. PCA
    # -------------------------------------------------
    with st.expander("🧠 PCA 기반 투자자 심리지수 구성"):

        st.markdown("""
        ATR, MFI, Stochastic residual은 서로 다른 성격을 가진 신호이기 때문에  
        일부 중복 정보와 노이즈를 포함할 가능성이 존재합니다.

        이에 따라 PCA(주성분 분석)를 활용하여 공통적으로 나타나는 흐름을  
        하나의 투자자 심리지수와 가까운 형태로 압축해볼 수 있는지 탐색하였습니다.

        이는 개별 지표를 제거하려는 목적이라기보다,  
        공통 심리 흐름에 가까운 특징을 비교적 안정적으로 살펴보기 위한 시도에 가깝습니다.
        """)

    # -------------------------------------------------
    # 5. XGBoost 선택 이유
    # -------------------------------------------------
    with st.expander("🤖 XGBoost 모델 선택 이유"):

        st.markdown("""
        본 프로젝트는 단순한 예측 정확도뿐 아니라  
        어떠한 조건과 조합이 예측에 기여했는지 함께 살펴보는 방향성을 고려하였습니다.

        XGBoost는 다음과 같은 특성이 연구 목적과 비교적 가까운 모델로 판단되었습니다.

        - 비선형 패턴을 반영할 가능성
        - 트리 분기(split)를 통한 복합 조건 탐색
        - 지표 간 상호작용(cross interaction) 반영 가능성
        - Ensemble 구조 기반 안정적 일반화 가능성

        예를 들어 Stochastic의 특정 구간과 변동성 신호가 함께 나타날 때  
        단순 선형관계보다 복합적인 심리 반응을 반영할 가능성을 추정해볼 수 있다고 보았습니다.
        """)

    # -------------------------------------------------
    # 6. 프로젝트 흐름
    # -------------------------------------------------
    with st.expander("🔄 프로젝트 분석 흐름", expanded=True):

        st.markdown("### 프로젝트 분석 Pipeline")

        c1, arrow1, c2, arrow2, c3, arrow3, c4, arrow4, c5, arrow5, c6 = st.columns(
            [2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2, 0.5, 2]
        )

        with c1:
            st.info("""
            **1️⃣ 데이터 수집**

            주가 데이터 확보  
            기간 및 종목 설정
            """)

        with arrow1:
            st.markdown("## ➡️")

        with c2:
            st.info("""
            **2️⃣ 기술지표 계산**

            ATR  
            MFI  
            STOCH
            """)

        with arrow2:
            st.markdown("## ➡️")

        with c3:
            st.info("""
            **3️⃣ Residual 추출**

            시장 일반 흐름 영향 일부 통제
            """)

        with arrow3:
            st.markdown("## ➡️")

        with c4:
            st.info("""
            **4️⃣ PCA 심리지수**

            공통 심리 흐름 압축
            """)

        with arrow4:
            st.markdown("## ➡️")

        with c5:
            st.info("""
            **5️⃣ XGBoost 학습**

            비선형 · 상호작용 학습
            """)

        with arrow5:
            st.markdown("## ➡️")

        with c6:
            st.success("""
            **6️⃣ 성능 검증**

            R² · RMSE  
            방향성 성공률
            """)

def render_data_prep_section(context):
    """
    조원들이 만든 all_might 데이터를 가져와서
    히트맵과 PCA 가중치를 시각화합니다.
    """

    st.markdown("## 📊 데이터 탐색 & 심리 proxy(PCA)")

    st.info("""
    본 탭은 데이터 전처리와 심리 proxy 구성 과정을 시각적으로 확인하는 영역입니다.

    데이터 수집 및 기술지표 계산은 `Samsung_crawling.py` 기준으로 수행되며,  
    ATR, MFI, Stochastic 지표를 바탕으로 residual을 추출한 뒤  
    PCA를 적용하여 `Investor_Sentiment_PC1`을 구성합니다.

    현재 시각화는 residual, PC1, 가격 변수 간 관계를 탐색적으로 확인하기 위한 EDA 성격이 강하며,  
    모델 성능을 직접적으로 설명하는 결과로 단정하지 않습니다.
    """)
    all_data = context["data"]


    for name, data in all_data.items():

        df = data[0]
        var_ratio = data[3]
        loadings = data[4]
        st.divider()   # 붙어남옴 방지!
        st.subheader(f"🔍 {name} 분석")

        # 히트맵 출력
        b64_hm = all_might.plot_heatmap(name, df)

        if b64_hm:
            st.write("### 잔차 상관계수 히트맵")
            st.caption(
                "이 히트맵은 모델 성능을 직접 설명하기보다는, "
                "residual과 PC1 및 가격 변수 간의 선형 관계를 "
                "탐색적으로 확인하기 위한 EDA 시각화입니다."
            )


            st.image(f"data:image/png;base64,{b64_hm}")

        # PCA 가중치
        st.write("### PCA 제1주성분 기여 가중치")
        
        st.metric(
        label="PC1 설명분산비율",
        value=f"{var_ratio * 100:.1f}%"
        )

        st.caption(
        "PC1이 ATR·MFI·STOCH residual의 공통 변동을 어느 정도 요약하는지 보여주는 값입니다."
        )
        

        col1, col2, col3 = st.columns(3)

        col1.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:32px; font-weight:bold;">ATR</div>
                <div style="font-size:32px;">{loadings[0]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col2.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:32px; font-weight:bold;">MFI</div>
                <div style="font-size:32px;">{loadings[1]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col3.markdown(
            f"""
            <div style="text-align:center;">
                <div style="font-size:32px; font-weight:bold;">STOCH</div>
                <div style="font-size:32px;">{loadings[2]:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------
        # PCA 자동 해석 카드
        # -------------------------------

        st.markdown("### 🧠 PCA 결과 해석")

        atr = abs(loadings[0])
        mfi = abs(loadings[1])
        stoch = abs(loadings[2])

        weights = {
            "ATR": atr,
            "MFI": mfi,
            "STOCH": stoch
        }

        sorted_weights = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top1 = sorted_weights[0][0]
        top2 = sorted_weights[1][0]

        st.success(f"""
        **해석 가이드**

        본 PCA 결과에서는 **{top1}**, **{top2}** 성분이
        상대적으로 크게 반영되었습니다.

        이는 투자자 심리와 가까운 공통 흐름이
        해당 지표에서 비교적 강하게 나타났을 가능성을 시사합니다.

        반면 기여도가 낮은 지표는 상대적으로
        공통 심리 흐름 설명력이 제한적일 수 있습니다.

        ※ PCA는 인과관계를 증명하는 과정이 아니라,
        공통 패턴을 탐색하기 위한 차원 축소 기법으로 해석합니다.
        """)

        st.divider()

# [섹션 구현: 모델 예측 결과]
def render_metric_section(context):
    st.markdown("## 🤖 모델 예측 성능")
    st.info("""
    **최종 모델 구조 요약**

    - **Model A**: Log Return lag 1~5 + Asset 더미 + Investor_Sentiment_PC1  
    - **Model B**: Log Return lag 1~5 + Asset 더미 + ATR/MFI/STOCH residual 3개  
    - **Model C**: Log Return lag 1~5 + Asset 더미 + Investor_Sentiment_PC1 + residual 3개  

    ※ pkl 파일 확인 결과, Model A/B/C는 각각 XGBRegressor 객체로 저장되어 있으며  
    입력 feature 수는 A=9개, B=11개, C=12개로 확인되었습니다.
    """)
    all_data = context["data"]
    
    for name, data in all_data.items():
        results = data[1] # 성능지표
        st.markdown("---")
        st.subheader(f"📈 {name}")
        st.caption("현재 표는 모델별 예측 성능을 비교하기 위한 시각화 영역입니다. "
                    "최종 모델 기준에서는 Price lag, Asset 더미, PC1, residual 조합별 설명력을 비교합니다."
                    )
        
        # 모델별 성능표(HTML 방식)를 가져와서 렌더링
        st.components.v1.html(all_might.metrics_table(results), height=300)

        # 2. [추가하면 좋을 부분] 표를 보고 판단하기 힘들 때 친절하게 알려주기
        # Model C의 성공률을 가져와서 판단하는 로직
        model_c = results.get("Model C (전체)")
        if model_c:
            acc = model_c.get("성공률(방향)", 0)
            if acc > 0.53:
                st.success(
                    f"📈 {name}: Model C가 테스트 구간에서 방향성을 일부 포착하는 경향을 보입니다. "
                    f"(방향성 성공률: {acc:.1%})"
                )
            elif acc > 0.50:
                st.info(
                    f"⚖️ {name}: 방향성 성공률이 50%를 소폭 상회합니다. "
                    f"추가 검증을 통해 안정성을 확인할 필요가 있습니다. "
                    f"(방향성 성공률: {acc:.1%})"
                )
            else:
                st.warning(
                    f"🔎 {name}: 테스트 구간에서는 방향성 예측력이 제한적으로 나타났습니다. "
                    f"추가 feature 조정 또는 검증 방식 개선이 필요합니다. "
                    f"(방향성 성공률: {acc:.1%})"
                )

def render_figure_section(context):
    all_data = context["data"]

    model_descriptions = {
        "Model A": "Investor Sentiment(PC1) 기반 모델",
        "Model B": "Residual 3개 기반 모델",
        "Model C": "Residual + PC1 결합 모델",
    }

    for name, data in all_data.items():
        pred_data = data[2]
        results = data[1]

        st.markdown("---")
        st.subheader(f"📉 {name} 예측 결과 시각화")
        st.caption( "각 모델의 실제 수익률과 예측 수익률 흐름을 마지막 테스트 구간 기준으로 비교합니다.")
        model_descriptions = {
        "Model A": "Price lag + Investor_Sentiment_PC1 기반 모델",
        "Model B": "Price lag + Residual 3개 기반 모델",
        "Model C": "Price lag + PC1 + Residual 3개 결합 모델",
        }
        for mname, (pred_s, true_s) in pred_data.items():
            st.markdown(f"### {mname} 예측 그래프")

            for key, description in model_descriptions.items():
                if key in mname:
                    st.caption(description)
                    break

            b64_plot = all_might.plot_pred(
                name,
                mname,
                true_s,
                pred_s,
                results[mname]
            )

            st.markdown(
                f"""
                <div style="
                    background-color:#fbf8f1;
                    border:1px solid #e3dacb;
                    border-radius:14px;
                    padding:14px;
                    margin:10px 0 34px 0;
                ">
                    <img src="data:image/png;base64,{b64_plot}"
                        style="width:100%; border-radius:10px; display:block;">
                </div>
                """,
                unsafe_allow_html=True
            )






# ==============================================================================
# 4. [탭 3] XGBoost 모델 예측 및 ML 지표 평가 영역 (권보성님/김민호님 기획 반영)
# ==============================================================================


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




# ==============================================================================
# 5. [탭 4] 시스템 인프라 및 자동화 영역 (기존 함수 유지)
# ==============================================================================

def render_log_section(context):
    """
    시스템 로그 및 프로젝트 정리 상태를 화면에 출력한다.

    이 영역은 실제 서버 로그라기보다는,
    발표용 대시보드에서 현재까지 확인된 파이프라인 구성과
    연결 상태를 요약하는 역할을 한다.
    """

    st.header("📜 시스템 로그")
    st.caption(
        "현재 대시보드 구성 상태와 최종 모델링 파이프라인 연결 기준을 요약합니다."
    )

    config = context.get("config", {})
    logs = context.get("logs", [])

    st.markdown("### ✅ 현재 확인된 기준")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"""
        **전처리 기준**

        {config.get("final_preprocessing", "확인 중")}
        """)

    with col2:
        st.info(f"""
        **모델링 기준**

        {config.get("final_modeling", "확인 중")}
        """)

    with col3:
        st.info(f"""
        **예측 타깃**

        {config.get("target", "확인 중")}
        """)

    st.markdown("---")
    st.markdown("### 🧾 작업 로그")

    if not logs:
        st.info("표시할 로그가 없습니다.")
        return

    for idx, log in enumerate(logs, start=1):
        st.write(f"**{idx}.** {log}")

    st.markdown("---")
    st.success(
        "현재 대시보드는 최종 모델링 결과 연결 전에도 발표 흐름과 시각화 구조를 확인할 수 있도록 구성되어 있습니다."
    )





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



if 'all_data' not in st.session_state:
    with st.spinner("조원들의 엔진(all_might.py)을 가동 중입니다..."):
        # 조원들의 모든 데이터를 한 번에 가져옵니다.
        all_data = {}
        for symbol, name in all_might.SYMBOLS.items():
            all_data[name] = all_might.process_symbol(symbol, name)
        st.session_state.all_data = all_data
        #st.session_state.data_loaded = True

context = {
    "config": {
        "app_title": APP_TITLE,
        "subtitle": APP_SUBTITLE,
        "model_source_note": MODEL_SOURCE_NOTE,

        # 현재 대시보드 상태
        "current_engine": "발표 시각화 참고용 구조",
        "final_preprocessing": "Samsung_crawling.py",
        "final_modeling": "train_pipeline.py + pkl 모델 A/B/C",

        # 최종 실험 기준
        "data_structure": "비트코인 · 삼성전자 · 코스피 패널 데이터",
        "target": "t+1 주봉 로그 수익률",
        "evaluation": "방향성 성공률, RMSE, R², 상관계수",
    },
    "data": st.session_state.all_data,
    "input": {
        "ticker": "비트코인 · 삼성전자 · 코스피",
        "period": "최근 6년 주봉",
        "window": "10주 기준",
    },
    "logs": [
        "Streamlit 대시보드 UI 구조 로드 완료",
        "데이터 전처리 기준 확인: Samsung_crawling.py",
        "최종 모델 구조 확인: XGBoost Model A/B/C",
        "pkl 모델 feature 구성 확인 완료",
        "그래프 시각화 카드형 레이아웃 적용 완료",
        "최종 모델링 결과는 train_pipeline.py 및 pkl 기준으로 연결 예정",
    ],
}

render_header(context)
render_sidebar(context)
tabs = render_tabs(context)

# 3. 탭별로 섹션 연결
with tabs[0]:
    render_overview_section(context)

with tabs[1]:
    # 조원들이 만든 잔차, PCA 차트가 여기서 출력됩니다.
    render_data_prep_section(context) 
with tabs[2]:
    # 모델 성능 지표 및 예측 결과 담기 (조원들의 XGBoost 예측 결과 출력)
    render_metric_section(context)
    # render_result_section(context)
    render_figure_section(context) 
    # render_figure_section(context) # 만약 그래프가 figures 딕셔너리에 있다면 사용
with tabs[3]:
    render_log_section(context)







