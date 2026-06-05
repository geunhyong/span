
from pathlib import Path
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from modules.data_fetcher import get_recent_data
from utils.data_utils import validate_columns

from visualization import plot_correlation_heatmap, plot_sentiment_index, plot_pca_loading_bar
from utils.table_utils import render_presentation_table
from utils.plotting_utils import new_figure, style_axis


INDICATOR_COLUMNS = ["ATR_10", "MFI_10", "STOCHk_10_3_3"]
RESIDUAL_COLUMNS = ["ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"]
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CACHE_DIR = Path("data/cache")
OUTPUT_DIR = Path("outputs")
DISPLAY_START_DATE = pd.Timestamp("2024-01-01")


def extract_sentiment_residuals(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["RET"] = result["Close"].pct_change()
    result["MOM"] = result["RET"].shift(1).rolling(window=10).sum()
    result["VOL"] = result["RET"].rolling(window=11).var()

    control_columns = ["RET", "MOM", "VOL"]
    clean = result.dropna(subset=control_columns + INDICATOR_COLUMNS).copy()
    controls = clean[control_columns]

    for indicator in INDICATOR_COLUMNS:
        model = LinearRegression()
        model.fit(controls, clean[indicator])
        clean[f"{indicator}_res"] = clean[indicator] - model.predict(controls)

    return clean


def create_sentiment_index(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    result = df.copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(result[RESIDUAL_COLUMNS])

    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(scaled).flatten()

    mfi_index = RESIDUAL_COLUMNS.index("MFI_10_res")
    if pca.components_[0][mfi_index] < 0:
        pc1 = -pc1

    result["Investor_Sentiment_PC1"] = pc1
    return result, float(pca.explained_variance_ratio_[0])


def _load_or_get_raw_data(asset_name: str) -> pd.DataFrame | None:
    """
    현재 선택 자산의 raw_data를 가져온다.
    데이터 전처리 탭에서 이미 불러온 데이터가 있으면 재사용하고,
    없으면 심리지표 탭에서 직접 불러올 수 있도록 한다.
    """
    raw_data_by_asset = st.session_state.get("raw_data_by_asset", {})

    if asset_name in raw_data_by_asset:
        df = raw_data_by_asset[asset_name]
        st.session_state.raw_data = df
        st.session_state.loaded_asset_name = asset_name
        return df

    raw_data = st.session_state.get("raw_data")
    loaded_asset_name = st.session_state.get("loaded_asset_name")

    if raw_data is not None and loaded_asset_name == asset_name:
        return raw_data

    return None


def _fetch_raw_data_for_sentiment(asset_name: str) -> pd.DataFrame:
    """
    심리지표 탭에서 선택 자산의 최근 주봉 데이터를 직접 불러온다.
    """
    df = get_recent_data(asset_name)
    validate_columns(df, REQUIRED_COLUMNS)

    if "raw_data_by_asset" not in st.session_state:
        st.session_state.raw_data_by_asset = {}

    st.session_state.raw_data_by_asset[asset_name] = df
    st.session_state.raw_data = df
    st.session_state.loaded_asset_name = asset_name

    return df

# CSV 저장/불러오기 함수 추가

def _safe_asset_name(asset_name: str) -> str:
    """
    파일명에 안전하게 사용할 자산명을 만든다.
    """
    return asset_name.replace("/", "_").replace(" ", "_")


def _get_sentiment_cache_path(asset_name: str) -> Path:
    """
    자산별 심리 proxy 계산 결과 CSV 경로를 반환한다.
    """
    return CACHE_DIR / f"{_safe_asset_name(asset_name)}_sentiment_proxy.csv"


def _get_sentiment_meta_path(asset_name: str) -> Path:
    """
    자산별 심리 proxy 계산 메타 정보 CSV 경로를 반환한다.
    """
    return CACHE_DIR / f"{_safe_asset_name(asset_name)}_sentiment_meta.csv"


def _filter_display_period(df: pd.DataFrame) -> pd.DataFrame:
    """
    발표 및 대시보드 표시용으로 2024년 이후 데이터만 남긴다.
    """
    result = df.copy()

    if not isinstance(result.index, pd.DatetimeIndex):
        result.index = pd.to_datetime(result.index)

    return result.loc[result.index >= DISPLAY_START_DATE].copy()


def _save_sentiment_cache(
    asset_name: str,
    sentiment_data: pd.DataFrame,
    explained_variance: float,
) -> None:
    """
    계산된 심리 proxy 결과를 CSV로 저장한다.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    save_df = _filter_display_period(sentiment_data)
    save_df.to_csv(
        _get_sentiment_cache_path(asset_name),
        encoding="utf-8-sig",
    )

    meta_df = pd.DataFrame(
        [
            {
                "항목": "explained_variance_pc1",
                "값": explained_variance,
            }
        ]
    )
    meta_df.to_csv(
        _get_sentiment_meta_path(asset_name),
        index=False,
        encoding="utf-8-sig",
    )


def _load_sentiment_cache(asset_name: str) -> tuple[pd.DataFrame | None, float | None]:
    """
    저장된 심리 proxy 계산 결과 CSV가 있으면 불러온다.
    """
    data_path = _get_sentiment_cache_path(asset_name)
    meta_path = _get_sentiment_meta_path(asset_name)

    if not data_path.exists() or not meta_path.exists():
        return None, None

    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    meta_df = pd.read_csv(meta_path)

    explained_variance = float(meta_df.loc[0, "값"])

    return df, explained_variance

# 월별 주봉 카드형 캘린더 함수 추가

### export  
def _load_pca_reference_csv() -> pd.DataFrame | None:
    """
    outputs/pca_reference.csv를 불러온다.
    기존 심리지표 계산 로직과 분리된 발표용 참고 시각화 데이터다.
    """
    path = OUTPUT_DIR / "pca_reference.csv"

    if not path.exists():
        return None

    return pd.read_csv(path)



########







def _format_calendar_value(value: float) -> str:
    """
    카드 안에 표시할 숫자를 짧게 정리한다.
    """
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "-"


def render_monthly_proxy_calendar(df: pd.DataFrame, start_year: int = 2026) -> None:
    """
    start_year 이후의 주봉 기준일별 심리 proxy 계산값을
    월별 카드 형태로 표시한다.

    캘린더 카드에는 PCA loading이 아니라,
    각 각 주봉 기준일별 PC1 및 residual 계산값을 표시한다.
    """
    if df.empty:
        st.info("표시할 심리 proxy 데이터가 없습니다.")
        return

    calendar_df = df.copy()

    if not isinstance(calendar_df.index, pd.DatetimeIndex):
        calendar_df.index = pd.to_datetime(calendar_df.index)

    calendar_df = calendar_df.loc[calendar_df.index.year >= start_year].copy()

    if calendar_df.empty:
        st.info(f"{start_year}년 이후 심리 proxy 데이터가 아직 없습니다.")
        return

    required_cols = [
        "Investor_Sentiment_PC1",
        "ATR_10_res",
        "MFI_10_res",
        "STOCHk_10_3_3_res",
    ]

    available_cols = [
        col for col in required_cols
        if col in calendar_df.columns
    ]

    if not available_cols:
        st.info("캘린더로 표시할 PC1 또는 residual 컬럼이 없습니다.")
        return

    st.subheader(f"{start_year}년 이후 월별 주봉 심리 proxy 카드")

    st.caption(
        f"아래 카드는 {start_year}년 이후 주봉 기준일별 심리 proxy 값을 월별로 정리한 것입니다. "
        f"각 주봉 기준일 카드에는 PC1과 residual 3개 값이 색상으로 구분되어 표시됩니다."
    )


    month_groups = list(
        calendar_df.groupby(
            [
                calendar_df.index.year,
                calendar_df.index.month,
            ],
            sort=True,
        )
    )

    cards_html = ""

    for (year, month), month_df in month_groups:
        month_df = month_df.sort_index()

        rows_html = ""

        for date, row in month_df.iterrows():
            date_text = date.strftime("%m-%d")

            pc1_value = _format_calendar_value(row.get("Investor_Sentiment_PC1", None))
            atr_value = _format_calendar_value(row.get("ATR_10_res", None))
            mfi_value = _format_calendar_value(row.get("MFI_10_res", None))
            stoch_value = _format_calendar_value(row.get("STOCHk_10_3_3_res", None))

            rows_html += textwrap.dedent(f"""
            <div class="proxy-week-row">
                <div class="proxy-week-date">{html.escape(date_text)} 주봉</div>
                <div class="proxy-badge-wrap">
                    <span class="proxy-badge proxy-pc1">PC1 {html.escape(pc1_value)}</span>
                    <span class="proxy-badge proxy-atr">ATR {html.escape(atr_value)}</span>
                    <span class="proxy-badge proxy-mfi">MFI {html.escape(mfi_value)}</span>
                    <span class="proxy-badge proxy-stoch">STOCHk {html.escape(stoch_value)}</span>
                </div>
            </div>
            """).strip() + "\n"

        month_card_html = textwrap.dedent(f"""
        <div class="proxy-month-card">
            <div class="proxy-month-title">{int(year)}년 {int(month)}월</div>
            {rows_html}
        </div>
        """).strip() + "\n"

        cards_html += month_card_html

    calendar_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: "Malgun Gothic", "Apple SD Gothic Neo", "Noto Sans KR", sans-serif;
                background: transparent;
            }}

            .proxy-wrapper {{
                width: 100%;
                box-sizing: border-box;
            }}

            .proxy-legend-wrap {{
                display: flex;
                flex-direction: column;
                gap: 8px;
                margin: 8px 0 18px 0;
                max-width: 260px;
            }}

            .proxy-legend-item {{
                display: inline-flex;
                align-items: center;
                gap: 7px;
                padding: 7px 11px;
                border-radius: 999px;
                background: #fbf8f1;
                border: 1px solid #e3dacb;
                font-size: 13px;
                font-weight: 700;
                color: #3d352d;
                width: fit-content;
            }}

            .legend-dot {{
                width: 11px;
                height: 11px;
                border-radius: 50%;
                display: inline-block;
            }}

            .pc1-dot {{ background: #5865f2; }}
            .atr-dot {{ background: #f97316; }}
            .mfi-dot {{ background: #16a34a; }}
            .stoch-dot {{ background: #db2777; }}

            .proxy-calendar-grid {{
                display: grid;
                grid-template-columns: repeat(3, minmax(260px, 1fr));
                gap: 20px;
                align-items: start;
                width: 100%;
                box-sizing: border-box;
            }}

            .proxy-month-card {{
                background: #fbf8f1;
                border: 1px solid #e3dacb;
                border-radius: 16px;
                padding: 14px 14px 12px 14px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
                box-sizing: border-box;
                width: 100%;
            }}

            .proxy-month-title {{
                font-size: 18px;
                font-weight: 850;
                color: #3d352d;
                margin-bottom: 10px;
            }}

            .proxy-week-row {{
                border-top: 1px solid #e7ddd0;
                padding: 9px 0 8px 0;
            }}

            .proxy-week-date {{
                font-size: 13px;
                font-weight: 800;
                color: #6e665e;
                margin-bottom: 7px;
            }}

            .proxy-badge-wrap {{
                display: flex;
                flex-direction: column;
                gap: 5px;
                align-items: flex-start;
            }}

            .proxy-badge {{
                display: inline-block;
                border-radius: 9px;
                padding: 4px 8px;
                font-size: 12px;
                font-weight: 750;
                color: #ffffff;
                line-height: 1.25;
                width: fit-content;
            }}

            .proxy-pc1 {{ background: #5865f2; }}
            .proxy-atr {{ background: #f97316; }}
            .proxy-mfi {{ background: #16a34a; }}
            .proxy-stoch {{ background: #db2777; }}
        </style>
    </head>
    <body>
        <div class="proxy-wrapper">
            <div class="proxy-legend-wrap">
                <div class="proxy-legend-item"><span class="legend-dot pc1-dot"></span>PC1</div>
                <div class="proxy-legend-item"><span class="legend-dot atr-dot"></span>ATR_10_res</div>
                <div class="proxy-legend-item"><span class="legend-dot mfi-dot"></span>MFI_10_res</div>
                <div class="proxy-legend-item"><span class="legend-dot stoch-dot"></span>STOCHk_10_3_3_res</div>
            </div>

            <div class="proxy-calendar-grid">
                {cards_html}
            </div>
        </div>
    </body>
    </html>
    """

    components.html(calendar_html, height=1800, scrolling=True)




def run() -> None:
    st.header("심리지표 계산 및 처리")

    st.info(
        "본 탭에서는 선택 자산의 ATR, MFI, Stochastic 지표에서 "
        "수익률·모멘텀·변동성으로 설명되는 부분을 OLS 회귀로 일부 통제한 뒤, "
        "남은 residual을 투자심리 proxy로 해석합니다. "
        "이후 residual 3개를 표준화하고 PCA를 적용하여 Investor_Sentiment_PC1을 생성합니다."
    )

    asset_name = st.session_state.get("asset_name", "삼성전자")

    raw_data = _load_or_get_raw_data(asset_name)

    if raw_data is None:
        st.warning(
            f"아직 {asset_name} 주봉 데이터가 준비되지 않았습니다. "
            "아래 버튼을 눌러 심리지표 계산에 사용할 데이터를 먼저 불러오세요."
        )

        if st.button(f"{asset_name} 심리지표용 주봉 데이터 불러오기", type="primary"):
            with st.spinner(f"{asset_name} 주봉 데이터를 불러오는 중입니다."):
                raw_data = _fetch_raw_data_for_sentiment(asset_name)
                st.success(f"{asset_name} 주봉 데이터가 준비되었습니다.")

        if raw_data is None:
            return

    if st.button(f"{asset_name} OLS 잔차 및 PCA 심리지수 계산", type="primary"):
        with st.spinner(f"{asset_name} 가격 요인을 통제한 잔차와 PCA 심리지수를 계산하는 중입니다."):
            residual_data = extract_sentiment_residuals(raw_data)
            sentiment_data, explained_variance = create_sentiment_index(residual_data)

            sentiment_data = _filter_display_period(sentiment_data)

            _save_sentiment_cache(
                asset_name=asset_name,
                sentiment_data=sentiment_data,
                explained_variance=explained_variance,
            )

            if "sentiment_data_by_asset" not in st.session_state:
                st.session_state.sentiment_data_by_asset = {}

            if "explained_variance_by_asset" not in st.session_state:
                st.session_state.explained_variance_by_asset = {}

            st.session_state.sentiment_data_by_asset[asset_name] = sentiment_data
            st.session_state.explained_variance_by_asset[asset_name] = explained_variance

            st.session_state.sentiment_data = sentiment_data
            st.session_state.explained_variance = explained_variance
            st.session_state.sentiment_asset_name = asset_name

            st.success(f"{asset_name} 심리 proxy 계산 결과를 CSV로 저장했습니다.")

    sentiment_data_by_asset = st.session_state.get("sentiment_data_by_asset", {})
    explained_variance_by_asset = st.session_state.get("explained_variance_by_asset", {})

    df = sentiment_data_by_asset.get(asset_name)
    explained_variance = explained_variance_by_asset.get(asset_name)

    if df is None or explained_variance is None:
        cached_df, cached_explained_variance = _load_sentiment_cache(asset_name)

        if cached_df is not None and cached_explained_variance is not None:
            df = cached_df
            explained_variance = cached_explained_variance

            if "sentiment_data_by_asset" not in st.session_state:
                st.session_state.sentiment_data_by_asset = {}

            if "explained_variance_by_asset" not in st.session_state:
                st.session_state.explained_variance_by_asset = {}

            st.session_state.sentiment_data_by_asset[asset_name] = df
            st.session_state.explained_variance_by_asset[asset_name] = explained_variance

            st.info(f"{asset_name} 심리 proxy 계산 결과를 저장된 CSV에서 불러왔습니다.")
        else:
            st.info(f"{asset_name} 심리지수 계산을 실행하세요.")
            return

    st.metric("PC1 설명분산", f"{explained_variance:.1%}")
    st.caption(
        "현재 표시되는 PC1 설명분산은 선택 자산 기준으로 새로 계산한 값입니다. "
        "최종 모델의 PC1 설명분산비율은 약 55.04%로 확인되었습니다."
    )

    pca_reference = pd.DataFrame(
        {
            "항목": [
                "PC1 설명분산비율",
                "ATR_10_res loading",
                "MFI_10_res loading",
                "STOCHk_10_3_3_res loading",
            ],
            "값": [
                "55.04%",
                "-0.0451",
                "0.7062",
                "0.7065",
            ],
            "해석": [
                "패널 학습 기준 residual 3개의 공통 변동 중 약 55%를 요약",
                "PC1 내 상대적 기여가 제한적으로 나타남",
                "PC1에 상대적으로 크게 반영됨",
                "PC1에 상대적으로 크게 반영됨",
            ],
        }
    )

    st.subheader("PCA 기준 참고값")
    render_presentation_table(
        pca_reference,
        title="PCA 기준 참고값",
        footnote=(
            "모델 생성 과정에서 확인한 PCA 기준 참고값입니다. "
            "화면의 PC1 설명분산은 선택 자산 기준으로 새로 계산되므로 최종 모델 기준값과 다를 수 있습니다."
        ),
        left_align_cols=["항목", "해석"],
    )

    #export 
    pca_reference_csv = _load_pca_reference_csv()

    if pca_reference_csv is not None:
        st.subheader("PCA loading 시각화")
        st.caption(
            "아래 그래프는 outputs/pca_reference.csv를 기반으로 한 발표용 PCA loading 시각화입니다. "
            "기존 심리지표 계산 로직은 변경하지 않습니다."
        )

        left, center, right = st.columns([0.10, 0.50, 0.10])

        with center:
            st.pyplot(
                plot_pca_loading_bar(pca_reference_csv),
                use_container_width=True,
            )
    else:
        st.info("outputs/pca_reference.csv가 없어 PCA loading 그래프를 표시하지 못했습니다.")
    ########

    st.caption(
        "PCA loading의 부호는 주성분 방향 설정에 따라 달라질 수 있으므로, "
        "절대적 방향성보다 상대적 기여도 중심으로 해석합니다."
    )
    with st.expander("심리 proxy값 데이터 table로 보기"):
        proxy_table_columns = [
            "Close",
            "ATR_10_res",
            "MFI_10_res",
            "STOCHk_10_3_3_res",
            "Investor_Sentiment_PC1",
        ]

        available_proxy_columns = [
            col for col in proxy_table_columns
            if col in df.columns
        ]

        proxy_display_df = df[available_proxy_columns].copy()
        proxy_display_df.index.name = "주봉 기준일"

        st.dataframe(
            proxy_display_df,
            use_container_width=True,
        )
    #render_monthly_proxy_calendar(df, start_year=2026)

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
    st.subheader("Investor Sentiment PC1 흐름")
    st.caption(
        "PC1은 residual 3개의 공통 흐름을 요약한 심리 proxy이며, "
        "양수와 음수 구간의 변화 방향을 시계열로 확인합니다."
    )
    st.pyplot(
        plot_sentiment_index(
            df["Investor_Sentiment_PC1"],
            title="Investor Sentiment PC1"
        ),
        use_container_width=True,
    )

    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
    st.subheader("Residual 및 PC1 상관관계")
    st.caption(
        "아래 heatmap은 ATR, MFI, Stochastic residual과 PC1 사이의 관계를 탐색적으로 확인하기 위한 시각화입니다. "
        "상관관계는 인과관계를 의미하지 않습니다."
    )
    st.pyplot(
        plot_correlation_heatmap(
            df[RESIDUAL_COLUMNS + ["Investor_Sentiment_PC1"]],
            title="Residual · PC1 Correlation"
        ),
        use_container_width=True,
    )
    
