"""
BTC/KRW 투자자 심리 지수 분석 파이프라인
데이터 수집 -> 기술 지표 계산 -> 시각화 + 저장 버튼 (Matplotlib)
"""

import sys
import logging
import numpy as np
import pandas as pd
import requests
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
from matplotlib import rcParams
from matplotlib.widgets import Button
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# 로깅 설정
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 전역 상수
# ──────────────────────────────────────────────
PALETTE = {
    'bg':    '#0f0f1a',
    'panel': '#12121f',
    'bull':  '#26a69a',
    'bear':  '#ef5350',
    'ma20':  '#ffa726',
    'ma60':  '#ab47bc',
    'bb':    '#636ef6',
    'rsi':   '#5c7cfa',
    'stk':   '#ffa726',
    'std':   '#ef5350',
    'atr':   '#ffa726',
    'obv':   '#636ef6',
    'text':  '#cccccc',
    'grid':  '#2a2a3a',
}

RSI_PERIOD   = 14
STOCH_PERIOD = 14
ATR_PERIOD   = 14
MA_SHORT     = 20
MA_LONG      = 60
PLOT_DAYS    = 180

SAVE_DPI = 150

# 한글 폰트 설정 (OS에 맞게 주석 전환)
rcParams['font.family'] = 'Malgun Gothic'   # Windows
# rcParams['font.family'] = 'AppleGothic'   # Mac
# rcParams['font.family'] = 'NanumGothic'   # Linux
rcParams['axes.unicode_minus'] = False
plt.style.use('dark_background')


# ──────────────────────────────────────────────
# 1. 데이터 수집
# ──────────────────────────────────────────────

def fetch_upbit_ohlcv(market: str = 'KRW-BTC', count: int = 500) -> pd.DataFrame:
    """업비트 일봉 OHLCV 수집 (최대 500일)"""
    all_data = []
    to = None
    headers = {"Accept": "application/json"}

    while len(all_data) < count:
        params = {"market": market, "count": min(200, count - len(all_data))}
        if to:
            params["to"] = to
        try:
            r = requests.get(
                "https://api.upbit.com/v1/candles/days",
                params=params, headers=headers, timeout=10
            )
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            log.error(f"업비트 API 오류: {e}")
            break

        if not data:
            break
        all_data.extend(data)
        to = data[-1]['candle_date_time_utc']

    df = pd.DataFrame(all_data)
    df['timestamp'] = pd.to_datetime(df['candle_date_time_kst'])
    df = df.rename(columns={
        'opening_price':           'open',
        'high_price':              'high',
        'low_price':               'low',
        'trade_price':             'close',
        'candle_acc_trade_volume': 'volume'
    })[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values('timestamp').set_index('timestamp').astype(float)
    return df


def fetch_fear_greed(limit: int = 500) -> pd.DataFrame:
    """Fear & Greed Index (alternative.me 무료 API)"""
    url = f'https://api.alternative.me/fng/?limit={limit}&format=json'
    try:
        raw = requests.get(url, timeout=10).json()
    except requests.RequestException as e:
        log.warning(f"FGI API 오류: {e}")
        return pd.DataFrame()

    fg = pd.DataFrame(raw['data'])
    fg['timestamp'] = pd.to_datetime(fg['timestamp'].astype(int), unit='s').dt.normalize()
    fg = fg.set_index('timestamp')[['value']].rename(columns={'value': 'FGI'})
    fg['FGI'] = fg['FGI'].astype(float)
    return fg


# ──────────────────────────────────────────────
# 2. 기술 지표 계산
# ──────────────────────────────────────────────

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, Stochastic, ATR, OBV, 볼린저밴드 계산"""

    # RSI
    delta = df['close'].diff()
    gain  = delta.clip(lower=0).rolling(RSI_PERIOD).mean()
    loss  = (-delta.clip(upper=0)).rolling(RSI_PERIOD).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))

    # Stochastic %K, %D
    low14  = df['low'].rolling(STOCH_PERIOD).min()
    high14 = df['high'].rolling(STOCH_PERIOD).max()
    df['Stoch_K'] = (df['close'] - low14) / (high14 - low14) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ATR (% 정규화)
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df['ATR']     = tr.rolling(ATR_PERIOD).mean()
    df['ATR_pct'] = df['ATR'] / df['close'] * 100

    # OBV — 벡터화 (루프 제거)
    sign = np.sign(df['close'].diff().fillna(0))
    df['OBV'] = (sign * df['volume']).cumsum()

    # 이동평균 & 볼린저밴드
    df['MA20']  = df['close'].rolling(MA_SHORT).mean()
    df['MA60']  = df['close'].rolling(MA_LONG).mean()
    df['BB_up'] = df['MA20'] + 2 * df['close'].rolling(MA_SHORT).std()
    df['BB_dn'] = df['MA20'] - 2 * df['close'].rolling(MA_SHORT).std()

    # 일간 수익률
    df['Return'] = df['close'].pct_change() * 100

    df.dropna(inplace=True)
    return df


# ──────────────────────────────────────────────
# 3. 공통 축 스타일 헬퍼
# ──────────────────────────────────────────────

def _style_ax(ax):
    """모든 패널에 동일한 다크 스타일 적용"""
    ax.set_facecolor(PALETTE['panel'])
    ax.tick_params(colors='#aaaaaa', labelsize=10)
    ax.grid(color=PALETTE['grid'], linewidth=0.5, alpha=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor(PALETTE['grid'])


# ──────────────────────────────────────────────
# 4. 대시보드 시각화 + 저장 버튼
# ──────────────────────────────────────────────

def plot_dashboard(df: pd.DataFrame, n: int = PLOT_DAYS):
    """4-패널 대시보드 + [PNG 저장] 버튼"""
    d = df.tail(n).copy()
    dates = d.index
    bg = PALETTE['bg']

    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor(bg)

    # 상단 버튼 공간 + 4개 패널
    gs = gridspec.GridSpec(
        5, 1,
        hspace=0.08,
        height_ratios=[0.18, 3, 1, 1, 1],
        top=0.95, bottom=0.07
    )

    ax_btn = fig.add_subplot(gs[0])
    ax1    = fig.add_subplot(gs[1])
    ax2    = fig.add_subplot(gs[2], sharex=ax1)
    ax3    = fig.add_subplot(gs[3], sharex=ax1)
    ax4    = fig.add_subplot(gs[4], sharex=ax1)

    ax_btn.set_visible(False)   # 버튼 배경 숨김

    for ax in [ax1, ax2, ax3, ax4]:
        _style_ax(ax)

    # ── Panel 1: 가격 + 볼린저밴드 + 캔들 ──
    ax1.fill_between(dates, d['BB_up'], d['BB_dn'],
                     alpha=0.07, color=PALETTE['bb'], label='볼린저밴드')
    ax1.plot(dates, d['BB_up'], color=PALETTE['bb'], linewidth=0.8, alpha=0.5)
    ax1.plot(dates, d['BB_dn'], color=PALETTE['bb'], linewidth=0.8, alpha=0.5)

    # 캔들 — 벡터화
    width  = pd.Timedelta(hours=14)
    colors = np.where(d['close'] >= d['open'], PALETTE['bull'], PALETTE['bear'])
    ax1.bar(
        dates,
        (d['close'] - d['open']).abs(),
        bottom=d[['open', 'close']].min(axis=1),
        width=width,
        color=colors,
        alpha=0.9
    )
    for idx, row in zip(dates, d[['low', 'high']].itertuples()):
        c = PALETTE['bull'] if d.loc[idx, 'close'] >= d.loc[idx, 'open'] else PALETTE['bear']
        ax1.plot([idx, idx], [row.low, row.high], color=c, linewidth=0.8)

    ax1.plot(dates, d['MA20'], color=PALETTE['ma20'],
             linewidth=1.4, linestyle='--', label=f'MA{MA_SHORT}', alpha=0.85)
    ax1.plot(dates, d['MA60'], color=PALETTE['ma60'],
             linewidth=1.4, linestyle=':', label=f'MA{MA_LONG}', alpha=0.85)
    ax1.set_ylabel('가격 (KRW)', color=PALETTE['text'], fontsize=11)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.0f}M'))
    ax1.legend(loc='upper left', fontsize=9,
               facecolor='#1a1a2e', edgecolor='#333', labelcolor=PALETTE['text'])
    ax1.set_title(
        f'BTC/KRW 투자자 심리 지표 대시보드 (최근 {n}일)',
        color='white', fontsize=14, pad=12, fontweight='bold'
    )
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Panel 2: RSI ──
    ax2.axhspan(70, 100, alpha=0.07, color=PALETTE['bear'])
    ax2.axhspan(0,  30,  alpha=0.07, color=PALETTE['bull'])
    ax2.axhline(70, color=PALETTE['bear'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.axhline(30, color=PALETTE['bull'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax2.axhline(50, color='#666666',       linewidth=0.6, linestyle=':',  alpha=0.5)
    ax2.plot(dates, d['RSI'], color=PALETTE['rsi'], linewidth=1.6, label='RSI(14)')
    ax2.fill_between(dates, d['RSI'], 50, where=d['RSI'] >= 50,
                     alpha=0.15, color=PALETTE['bear'])
    ax2.fill_between(dates, d['RSI'], 50, where=d['RSI'] <  50,
                     alpha=0.15, color=PALETTE['bull'])
    ax2.set_ylim(0, 100)
    ax2.set_yticks([30, 50, 70])
    ax2.set_ylabel('RSI', color=PALETTE['text'], fontsize=10)
    ax2.text(0.01, 0.85, '과매수', transform=ax2.transAxes,
             color=PALETTE['bear'], fontsize=8, alpha=0.8)
    ax2.text(0.01, 0.05, '과매도', transform=ax2.transAxes,
             color=PALETTE['bull'], fontsize=8, alpha=0.8)
    plt.setp(ax2.get_xticklabels(), visible=False)

    # ── Panel 3: Stochastic ──
    ax3.axhspan(80, 100, alpha=0.07, color=PALETTE['bear'])
    ax3.axhspan(0,  20,  alpha=0.07, color=PALETTE['bull'])
    ax3.axhline(80, color=PALETTE['bear'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax3.axhline(20, color=PALETTE['bull'], linewidth=0.8, linestyle='--', alpha=0.6)
    ax3.plot(dates, d['Stoch_K'], color=PALETTE['stk'],  linewidth=1.6, label='%K')
    ax3.plot(dates, d['Stoch_D'], color=PALETTE['std'],  linewidth=1.2,
             linestyle='--', label='%D', alpha=0.8)
    ax3.set_ylim(0, 100)
    ax3.set_yticks([20, 50, 80])
    ax3.set_ylabel('Stochastic', color=PALETTE['text'], fontsize=10)
    ax3.legend(loc='upper right', fontsize=8,
               facecolor='#1a1a2e', edgecolor='#333', labelcolor=PALETTE['text'])
    plt.setp(ax3.get_xticklabels(), visible=False)

    # ── Panel 4: ATR% ──
    avg_atr = d['ATR_pct'].mean()
    ax4.fill_between(dates, d['ATR_pct'], alpha=0.3, color=PALETTE['atr'])
    ax4.plot(dates, d['ATR_pct'], color=PALETTE['atr'], linewidth=1.6, label='ATR%')
    ax4.axhline(avg_atr, color='white', linewidth=0.8, linestyle='--',
                alpha=0.5, label=f'평균 {avg_atr:.2f}%')
    ax4.set_ylabel('ATR%', color=PALETTE['text'], fontsize=10)
    ax4.legend(loc='upper right', fontsize=8,
               facecolor='#1a1a2e', edgecolor='#333', labelcolor=PALETTE['text'])
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax4.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax4.get_xticklabels(), rotation=0, ha='center', color='#aaaaaa')
    ax4.set_xlabel('날짜', color=PALETTE['text'], fontsize=11)

    # ── 저장 버튼 ──
    btn_ax = fig.add_axes([0.82, 0.965, 0.12, 0.028])
    btn = Button(btn_ax, '💾  PNG 저장', color='#1e1e30', hovercolor='#2e2e50')
    btn.label.set_color('white')
    btn.label.set_fontsize(10)

    out_path = 'btc_dashboard.png'

    def on_save(_event):
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor=bg)
        log.info(f"대시보드 저장 완료: {out_path}")
        btn.label.set_text('✅ 저장됨!')
        fig.canvas.draw_idle()

    btn.on_clicked(on_save)

    plt.show()
    return fig


# ──────────────────────────────────────────────
# 5. OBV 차트 + 저장 버튼
# ──────────────────────────────────────────────

def plot_obv(df: pd.DataFrame, n: int = PLOT_DAYS):
    """OBV 단독 차트 + [PNG 저장] 버튼"""
    d = df.tail(n).copy()
    dates = d.index
    bg = PALETTE['bg']

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.subplots_adjust(bottom=0.15, top=0.88)
    fig.patch.set_facecolor(bg)
    _style_ax(ax)

    ax.fill_between(dates, d['OBV'], alpha=0.15, color=PALETTE['obv'])
    ax.plot(dates, d['OBV'], color=PALETTE['obv'], linewidth=1.8, label='OBV')
    ax.plot(
        dates,
        d['OBV'].rolling(MA_SHORT).mean(),
        color=PALETTE['ma20'],
        linewidth=1.4,
        linestyle='--',
        label=f'OBV MA{MA_SHORT}'
    )
    ax.set_title('BTC OBV — 거래량 누적 심리 지수',
                 color='white', fontsize=13, fontweight='bold')
    ax.set_ylabel('OBV (누적)', color=PALETTE['text'])
    ax.set_xlabel('날짜', color=PALETTE['text'])
    ax.tick_params(colors='#aaaaaa')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#333',
              labelcolor=PALETTE['text'])

    # ── 저장 버튼 ──
    btn_ax = fig.add_axes([0.82, 0.92, 0.13, 0.06])
    btn = Button(btn_ax, '💾  PNG 저장', color='#1e1e30', hovercolor='#2e2e50')
    btn.label.set_color('white')
    btn.label.set_fontsize(10)

    out_path = 'btc_obv.png'

    def on_save(_event):
        fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches='tight', facecolor=bg)
        log.info(f"OBV 차트 저장 완료: {out_path}")
        btn.label.set_text('✅ 저장됨!')
        fig.canvas.draw_idle()

    btn.on_clicked(on_save)

    plt.show()
    return fig


# ──────────────────────────────────────────────
# 6. 메인 실행
# ──────────────────────────────────────────────

def main():
    log.info("=" * 50)
    log.info("BTC/KRW 투자자 심리 분석 파이프라인")
    log.info("=" * 50)

    # 1) OHLCV 수집
    log.info("[1/4] 업비트 OHLCV 수집 중...")
    df = fetch_upbit_ohlcv(market='KRW-BTC', count=500)
    log.info(f"      → {len(df)}일치 데이터 수집 완료")

    # 2) 지표 계산
    log.info("[2/4] 기술 지표 계산 중...")
    df = calc_indicators(df)
    log.info(f"      → 지표 계산 완료 (유효 데이터: {len(df)}일)")

    # 3) CSV 저장
    log.info("[3/4] CSV 저장 중...")
    df.to_csv('btc_indicators.csv')
    log.info("      → btc_indicators.csv 저장 완료")

    # 4) 시각화 (창이 열리면 버튼으로 저장)
    log.info("[4/4] 시각화 창을 엽니다. 창 상단의 [💾 PNG 저장] 버튼을 클릭하세요.")
    plot_dashboard(df, n=PLOT_DAYS)
    plot_obv(df, n=PLOT_DAYS)

    # 5) 최신 수치 출력
    latest = df.iloc[-1]
    log.info("=" * 50)
    log.info("[최신 지표값]")
    log.info(f"  날짜     : {df.index[-1].strftime('%Y-%m-%d')}")
    log.info(f"  종가     : {latest['close']:,.0f} KRW")
    log.info(f"  RSI      : {latest['RSI']:.1f}")
    log.info(f"  Stoch %K : {latest['Stoch_K']:.1f}")
    log.info(f"  ATR%     : {latest['ATR_pct']:.2f}%")
    log.info(f"  OBV      : {latest['OBV']:,.0f}")
    log.info("=" * 50)


if __name__ == '__main__':
    main()
