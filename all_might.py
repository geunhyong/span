# -*- coding: utf-8 -*-
"""
all_might.py
비트코인(BTC-USD) · 삼성전자(005930.KS) · 코스닥(^KQ11)
5년 주봉 → OLS 잔차 → PCA(투자자 심리지수) → XGBoost
  Model A : Close + PC1          (통합 심리지수)
  Model B : Close + 잔차 3개      (ATR·MFI·STOCH 개별)
  Model C : Close + PC1 + 잔차 3개 (전체 혼합)
→ 성공률(방향 정확도) · 상관계수 · R²/RMSE/MAE → report.html
"""

import io, base64, datetime, platform, webbrowser, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
#  0. 전역 설정
# ═══════════════════════════════════════════════════════════════
SYMBOLS = {
    "BTC-USD":   "비트코인",
    "005930.KS": "삼성전자",
    "^KQ11":     "코스닥",
}
END_DATE     = pd.Timestamp("2026-05-25")
START_DATE   = END_DATE - pd.DateOffset(years=5)
# WARMUP_START: 분석 시작일(START_DATE)보다 20주 앞에서부터 데이터를 받는 이유 →
#   MFI(10주)·STOCH(10주)·Volatility(5주) 같은 rolling 지표는 윈도우가 채워지기 전까지
#   NaN이 나온다. START_DATE 당일부터 지표가 '값이 있는' 상태로 시작하게 하려면
#   그만큼의 선행(warm-up) 구간이 필요하다. 10주 윈도우 + 평활/여유분 = 20주.
WARMUP_START = START_DATE - pd.DateOffset(weeks=20)

# 삼성전자 발행주식수(자사주 소각 반영)를 '변경일 → 주식수'로 직접 명시한다.
#   왜 하드코딩? yfinance가 과거 발행주식수 시계열을 항상 안정적으로 주지 않기 때문.
#   변경 시점만 적어두면 아래 get_shares()의 ffill로 구간 전체에 자동 전파된다.
#   (단점: 주식수가 또 바뀌면 여기 날짜/수치를 수동으로 갱신해야 함)
SAMSUNG_SHARES = pd.Series({
    pd.Timestamp("2020-01-01"): 5_969_782_550.0,   # 기준 시작값
    pd.Timestamp("2025-03-26"): 5_919_637_922.0,   # 자사주 소각 등으로 감소
    pd.Timestamp("2026-04-14"): 5_846_278_608.0,   # 추가 감소
})

N_SPLITS = 5
RES_COLS  = ["ATR_res", "MFI_res", "STOCH_res"]
XGB_PARAMS = dict(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbosity=0,
)

if platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
elif platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
else:
    plt.rc("font", family="DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False

COLORS = {"red": "#9d3326", "green": "#2f6f5e", "gold": "#b07d27",
          "ink": "#26201c", "bg": "#f9f6f0", "bg2": "#efe9dd", "line": "#d8cfbe"}

# ═══════════════════════════════════════════════════════════════
#  1. 데이터 수집
# ═══════════════════════════════════════════════════════════════
def fetch_weekly(symbol: str) -> pd.DataFrame:
    raw = yf.download(
        symbol,
        start=WARMUP_START.strftime("%Y-%m-%d"),
        # yfinance의 end는 '미포함(exclusive)'이라 END_DATE 주를 놓칠 수 있다.
        # 주봉(1wk)이라 하루가 아닌 +7일을 더해 마지막 주가 확실히 포함되게 한다.
        end=(END_DATE + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
        # auto_adjust=True: 배당·액면분할이 반영된 '수정주가'를 받는다.
        #   안 하면 분할 시점에 가격이 점프해서 수익률/지표가 왜곡된다.
        interval="1wk", auto_adjust=True, progress=False,
    )
    if raw.empty:
        raise ValueError(f"데이터 없음: {symbol}")
    # 최신 yfinance는 단일 티커도 (지표, 티커) 2단 컬럼(MultiIndex)으로 줄 때가 있다.
    # 0번째 레벨(Open/High/...)만 남겨 평범한 단일 컬럼으로 평탄화한다.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    # 야후 인덱스에 붙은 타임존을 제거(tz-naive)한다. START/END_DATE도 tz가 없으므로,
    # 비교(df.index <= END_DATE) 시 'tz-aware vs naive' 에러를 피하려면 통일이 필요.
    df.index = pd.to_datetime(df.index).tz_localize(None)
    # 워밍업 때문에 END_DATE 이후 주가 섞여 들어올 수 있어 잘라낸다.
    # dropna는 Close만 기준 → 지수(^KQ11 등)는 Volume=0/NaN이 정상이라 OHLC 전체로
    # 거르면 멀쩡한 행까지 날아간다. 가격의 핵심인 Close만 있으면 유효 행으로 본다.
    df = df[df.index <= END_DATE].dropna(subset=["Close"])
    # Return: 전주 대비 수익률(소수 단위, 0.01 = 1%). %로 바꾸지 않는 이유는
    #   타깃·모멘텀·변동성 계산이 모두 이 소수 스케일에 맞춰져 있기 때문.
    df["Return"] = df["Close"].pct_change()
    print(f"  수집: {len(df)}주  ({df.index[0].date()} ~ {df.index[-1].date()})")
    return df

# ═══════════════════════════════════════════════════════════════
#  2. 기술 지표 계산
# ═══════════════════════════════════════════════════════════════
def calc_mfi(df: pd.DataFrame, length: int = 10) -> pd.Series:
    # 거래량 0은 '거래 없음'이지 '돈 흐름 0'이 아니다. 0으로 두면 그 주의 자금흐름이
    # 통째로 0 처리되어 MFI가 왜곡되므로 NaN으로 빼서 계산에서 제외한다.
    vol = df["Volume"].replace(0, np.nan)
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3   # 대표가격(typical price)
    mf  = tp * vol                                      # 자금흐름(money flow)
    # tp가 전주보다 오른 주의 자금흐름만 pos, 내린 주만 neg로 분리해 length주간 합산.
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(length, min_periods=length).sum()
    # neg=0(하락 주가 전혀 없음)이면 0으로 나누게 되므로 NaN 처리 → 분모 0 방지.
    mfr = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + mfr)).rename("MFI_10")

def calc_stoch_k(df: pd.DataFrame, lk: int = 10, sk: int = 3) -> pd.Series:
    lo = df["Low"].rolling(lk, min_periods=lk).min()    # 최근 lk주 최저가
    hi = df["High"].rolling(lk, min_periods=lk).max()   # 최근 lk주 최고가
    # hi==lo(변동 폭 0)면 0으로 나누므로 NaN 처리. raw %K를 sk주 평활해 노이즈를 줄임.
    raw_k = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return raw_k.rolling(sk, min_periods=sk).mean().rename("STOCHk_10_3")

def calc_dir_atr(df: pd.DataFrame, shares: pd.Series) -> pd.Series:
    # 방향성 회전율(ATR): 회전율에 그 주 수익률의 '부호'를 곱해 매수/매도 압력의
    # 방향까지 담는다. 수익률이 정확히 0이거나 NaN이면 방향을 정의할 수 없어 NaN.
    sign = df["Return"].apply(
        lambda r: np.nan if (pd.isna(r) or r == 0) else float(np.sign(r))
    )
    # 회전율 = 거래량 / 발행주식수. Volume=0, shares=0/NaN은 모두 NaN으로 막아
    # 0 나눗셈을 방지. ×1000은 회전율이 작은 값이라 보기 좋은 스케일로 키우는 것.
    turnover = df["Volume"].replace(0, np.nan) / shares.replace(0, np.nan)
    return (turnover * sign * 1000).rename("ATR_10")

def calc_std_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean().rename("ATR_10")

def get_shares(symbol: str, idx: pd.DatetimeIndex) -> pd.Series:
    # 삼성전자: 위에서 정의한 변경일 기준 주식수를 주봉 날짜에 매핑한다.
    #   reindex(method="ffill") → 변경일 이후 구간은 직전 값을 그대로 유지(전파).
    #   bfill() → 가장 이른 주봉이 첫 변경일보다 앞서면 가장 가까운 이후 값으로 메움.
    if symbol == "005930.KS":
        s = SAMSUNG_SHARES.sort_index()
        return s.reindex(idx, method="ffill").bfill()
    # 그 외(비트코인 등): 발행주식수 개념이 없으므로 유통량(circulatingSupply)을 대체.
    #   info 호출은 실패할 수 있어 try로 감싸고, 없으면 전 구간 NaN → 표준 ATR로 우회.
    try:
        val = yf.Ticker(symbol).info.get("circulatingSupply", 0)
        if val and val > 0:
            return pd.Series(float(val), index=idx)
    except Exception:
        pass
    return pd.Series(np.nan, index=idx)

def build_indicators(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    df["MFI_10"]      = calc_mfi(df)
    df["STOCHk_10_3"] = calc_stoch_k(df)
    # 지수(^KQ11 등)는 '발행주식수'가 없어 회전율 기반 ATR을 못 만든다.
    # → 종목은 방향성 회전율 ATR, 지수는 표준 ATR(Wilder)로 분기한다.
    if symbol.startswith("^"):
        df["ATR_10"] = calc_std_atr(df)
    else:
        shares = get_shares(symbol, df.index)
        df["ATR_10"] = calc_dir_atr(df, shares)
    return df

# ═══════════════════════════════════════════════════════════════
#  3. OLS 잔차 추출 (시장 일반 요인 제거)
# ═══════════════════════════════════════════════════════════════
def ols_residual(y: pd.Series, Xdf: pd.DataFrame) -> pd.Series:
    # 지표(y)를 가격 요인(X: 수익률·모멘텀·변동성)으로 회귀한 뒤 '잔차'만 남긴다.
    # 잔차 = 가격으로 설명되지 않는 부분 → 가격과 무관한 '심리 proxy에 가까운 성분'으로 해석.
    # mask: y나 X 중 하나라도 NaN인 행은 회귀에서 제외(LinearRegression은 NaN 불가).
    mask = y.notna() & Xdf.notna().all(axis=1)
    out  = pd.Series(np.nan, index=y.index, name=y.name)
    if mask.sum() < 20:   # 표본이 너무 적으면 회귀 신뢰 불가 → 전부 NaN 반환
        return out
    m = LinearRegression()
    m.fit(Xdf[mask].values, y[mask].values)
    out[mask] = y[mask].values - m.predict(Xdf[mask].values)   # 실제값 - 예측값
    return out

def calc_residuals(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["Return"]
    # MOM에 shift(1): '직전 주까지'의 누적 수익률만 본다. shift 없이 당주를 포함하면
    #   같은 주 정보가 통제변수로 새어 들어가 잔차 추출이 부정확해진다.
    mom = ret.rolling(10, min_periods=10).sum().shift(1)   # 직전 10주 누적수익률(모멘텀)
    vol = ret.rolling(10, min_periods=10).var()            # 10주 수익률 분산(변동성)
    Xdf = pd.DataFrame({"RET": ret, "MOM": mom, "VOL": vol})

    df["ATR_res"]   = ols_residual(df["ATR_10"],      Xdf)
    df["MFI_res"]   = ols_residual(df["MFI_10"],      Xdf)
    df["STOCH_res"] = ols_residual(df["STOCHk_10_3"], Xdf)
    return df

# ═══════════════════════════════════════════════════════════════
#  4. PCA → 투자자 심리지수 (PC1)
# ═══════════════════════════════════════════════════════════════
def calc_pca(df: pd.DataFrame):
    res_cols = ["ATR_res", "MFI_res", "STOCH_res"]
    sub = df[res_cols].dropna()
    df["PC1"] = np.nan
    if len(sub) < 15:
        print("  PCA: 유효 행 부족 → PC1 NaN")
        return df, 0.0, np.zeros(3)
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(sub.values)
    pca     = PCA(n_components=1, random_state=42)
    pc1_vals = pca.fit_transform(scaled).flatten()
    df.loc[sub.index, "PC1"] = pc1_vals
    var_ratio = float(pca.explained_variance_ratio_[0])
    loadings  = pca.components_[0]
    print(f"  PCA: PC1 설명분산 {var_ratio:.1%}  "
          f"loadings ATR={loadings[0]:.2f} MFI={loadings[1]:.2f} STOCH={loadings[2]:.2f}")
    return df, var_ratio, loadings

# ═══════════════════════════════════════════════════════════════
#  5. 피처 빌드
# ═══════════════════════════════════════════════════════════════
PRICE_FEATS = ["Return_lag1", "Return_lag2", "Return_lag4", "Volatility_5w"]
MODEL_FEATS = {
    "Model A (PC1)":     PRICE_FEATS + ["PC1"],
    "Model B (잔차3개)": PRICE_FEATS + RES_COLS,
    "Model C (전체)":    PRICE_FEATS + ["PC1"] + RES_COLS,
}

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Target = shift(-1): '다음 주' 수익률을 정답으로 끌어온다. 모델은 현재 시점 피처로
    #   미래(t+1)를 맞히는 것이므로 타깃만 한 칸 당겨야 한다.
    d["Target"]       = d["Return"].shift(-1)               # t+1 주봉 수익률 (예측 대상)
    # 피처는 모두 '현재 시점에서 이미 알 수 있는' 과거 값이어야 미래 누출이 없다.
    d["Return_lag1"]  = d["Return"]                          # 당주 수익률
    d["Return_lag2"]  = d["Return"].shift(1)                 # 전주 수익률
    d["Return_lag4"]  = d["Return"].shift(3)                 # 3주 전 수익률
    # Volatility도 shift(1): 당주 변동성을 쓰면 당주 종가(=거의 타깃 직전)가 새어 들어옴.
    d["Volatility_5w"] = d["Return"].rolling(5, min_periods=5).std().shift(1)
    return d

# ═══════════════════════════════════════════════════════════════
#  6. 모델 학습 및 평가
# ═══════════════════════════════════════════════════════════════
def direction_accuracy(y_true, y_pred):
    # 타겟이 수익률이므로 부호 자체가 방향
    act   = np.sign(np.asarray(y_true))
    prd   = np.sign(np.asarray(y_pred))
    valid = act != 0
    if valid.sum() == 0:
        return np.nan
    return float((act[valid] == prd[valid]).mean())

def evaluate_all(y_true, y_pred):
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    return {
        "R²":           r2_score(yt, yp),
        "RMSE(%)":      float(np.sqrt(mean_squared_error(yt, yp))) * 100,
        "MAE(%)":       float(mean_absolute_error(yt, yp)) * 100,
        "상관계수":     float(np.corrcoef(yt, yp)[0, 1]),
        "성공률(방향)": direction_accuracy(yt, yp),
    }

def run_cv(df: pd.DataFrame, feat_cols: list):
    use_pca = "PC1" in feat_cols
    use_res = any(c in feat_cols for c in RES_COLS)
    base_feats = [c for c in feat_cols if c not in ["PC1"] + RES_COLS]

    need_cols = base_feats + RES_COLS + ["Target"]
    data = df[[c for c in need_cols if c in df.columns]].dropna()
    if len(data) < 30:
        return None, None, None

    X_base = data[base_feats].values
    X_res  = data[RES_COLS].values
    y      = data["Target"].values

    # 시계열 데이터라 무작위 분할(KFold)은 금물 → 과거로 학습해 미래를 검증해야 한다.
    # TimeSeriesSplit은 항상 '훈련 구간 < 테스트 구간' 순서를 지켜 누출을 막는다.
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    all_true, all_pred = [], []
    last_pred_arr = last_true_arr = last_idx = None
    # 과적합 점검용: 마지막(가장 큰) fold의 '훈련 구간' 예측을 따로 보관.
    # 훈련 성공률과 테스트 성공률을 비교해 둘이 벌어지면 과적합 신호.
    last_tr_pred = last_tr_true = None

    for tr_idx, te_idx in tscv.split(X_base):
        if len(te_idx) < 3:
            continue

        Xtr_list = [X_base[tr_idx]]
        Xte_list = [X_base[te_idx]]

        if use_pca:
            # 누출 방지의 핵심: Scaler·PCA를 '훈련 fold'로만 fit하고, 테스트 fold에는
            #   transform만 적용한다. 전체 데이터로 미리 fit하면 테스트(미래) 정보가
            #   평균/분산/주성분 축에 섞여 들어가 성능이 부풀려진다.
            sc  = StandardScaler()
            pca = PCA(n_components=1, random_state=42)
            Xtr_list.append(pca.fit_transform(sc.fit_transform(X_res[tr_idx])))
            Xte_list.append(pca.transform(sc.transform(X_res[te_idx])))

        if use_res:
            Xtr_list.append(X_res[tr_idx])
            Xte_list.append(X_res[te_idx])

        Xtr = np.hstack(Xtr_list)
        Xte = np.hstack(Xte_list)

        mdl = XGBRegressor(**XGB_PARAMS)
        mdl.fit(Xtr, y[tr_idx])
        pred       = mdl.predict(Xte)
        pred_train = mdl.predict(Xtr)   # 같은 모델로 훈련 구간도 예측(과적합 점검용)

        all_true.extend(y[te_idx])
        all_pred.extend(pred)
        last_pred_arr, last_true_arr, last_idx = pred, y[te_idx], te_idx
        last_tr_pred, last_tr_true = pred_train, y[tr_idx]

    if not all_true or last_idx is None:
        return None, None, None

    metrics = evaluate_all(all_true, all_pred)
    # 훈련 성공률(방향)을 metrics에 추가 → 콘솔/HTML 표에 테스트와 나란히 표시.
    # 테스트 성공률보다 크게 높으면 과적합(트리 수↓·max_depth↓ 검토).
    metrics["훈련 성공률"] = direction_accuracy(last_tr_true, last_tr_pred)
    pred_s  = pd.Series(last_pred_arr, index=data.index[last_idx], name="Predicted")
    true_s  = pd.Series(last_true_arr, index=data.index[last_idx], name="Actual")
    return metrics, pred_s, true_s

# ═══════════════════════════════════════════════════════════════
#  7. 시각화 헬퍼
# ═══════════════════════════════════════════════════════════════
def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def plot_pred(name: str, model_name: str,
              true_s: pd.Series, pred_s: pd.Series, metrics: dict) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(f"{name}  |  {model_name}", fontsize=12, fontweight="bold",
                 color=COLORS["ink"])

    for ax in (ax1, ax2):
        ax.set_facecolor(COLORS["bg"])
        for sp in ax.spines.values():
            sp.set_color(COLORS["line"])

    ax1.plot(true_s.index, true_s.values * 100, color=COLORS["ink"],  lw=1.6, label="실제 수익률")
    ax1.plot(pred_s.index, pred_s.values * 100, color=COLORS["red"],  lw=1.6,
             linestyle="--", label="예측 수익률")
    ax1.axhline(0, color="#aaa", lw=0.7, linestyle=":")
    ax1.set_ylabel("%", fontsize=8)
    ax1.set_title("실제 vs 예측 수익률 (마지막 Test Fold)", fontsize=10)
    ax1.legend(fontsize=9); ax1.tick_params(labelsize=8)

    tv, pv = true_s.values * 100, pred_s.values * 100
    mn, mx = min(tv.min(), pv.min()), max(tv.max(), pv.max())
    ax2.scatter(tv, pv, alpha=0.55, s=22,
                color=COLORS["green"], edgecolors="none")
    ax2.plot([mn, mx], [mn, mx], "k--", lw=0.9, alpha=0.45)
    ax2.axhline(0, color="#aaa", lw=0.5); ax2.axvline(0, color="#aaa", lw=0.5)
    ax2.set_xlabel("실제 수익률(%)", fontsize=9); ax2.set_ylabel("예측 수익률(%)", fontsize=9)
    corr = metrics.get("상관계수", 0)
    acc  = metrics.get("성공률(방향)", 0)
    ax2.set_title(f"산점도  |  상관계수 {corr:.3f}  성공률 {acc:.1%}", fontsize=10)
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    return fig_b64(fig)

def plot_residuals_and_pc1(name: str, df: pd.DataFrame, var_ratio: float,
                           loadings: np.ndarray):
    # 잔차 3개 막대
    fig1, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    fig1.patch.set_facecolor(COLORS["bg"])
    fig1.suptitle(f"{name}  |  OLS 잔차 (심리 proxy에 가까운 성분)", fontsize=11,
                  fontweight="bold", color=COLORS["ink"])
    pairs = [("ATR_res","ATR 잔차",COLORS["red"]),
             ("MFI_res","MFI 잔차",COLORS["green"]),
             ("STOCH_res","STOCH 잔차",COLORS["gold"])]
    for ax, (col, lbl, c) in zip(axes, pairs):
        s = df[col].dropna()
        ax.bar(s.index, s.values, color=c, alpha=0.75, width=5)
        ax.axhline(0, color="#888", lw=0.7)
        ax.set_ylabel(lbl, fontsize=8)
        ax.set_facecolor(COLORS["bg"])
        for sp in ax.spines.values(): sp.set_color(COLORS["line"])
        ax.tick_params(labelsize=7)
    fig1.tight_layout()
    b64_res = fig_b64(fig1)

    # PC1 시계열
    fig2, ax = plt.subplots(figsize=(11, 2.8))
    fig2.patch.set_facecolor(COLORS["bg"]); ax.set_facecolor(COLORS["bg"])
    for sp in ax.spines.values(): sp.set_color(COLORS["line"])
    pc1 = df["PC1"].dropna()
    ax.fill_between(pc1.index, pc1.values, 0,
                    where=pc1.values >= 0, color=COLORS["green"], alpha=0.65, label="상대적 양(+)의 심리 구간(낙관)")
    ax.fill_between(pc1.index, pc1.values, 0,
                    where=pc1.values  < 0, color=COLORS["red"],   alpha=0.65, label="상대적 음(-)의 심리 구간(공포)")
    ax.axhline(0, color="#888", lw=0.8)
    load_str = f"ATR {loadings[0]:+.2f}  MFI {loadings[1]:+.2f}  STOCH {loadings[2]:+.2f}"
    ax.set_title(f"{name}  |  투자자 심리지수 PC1  (설명분산 {var_ratio:.1%})  "
                 f"loadings: {load_str}", fontsize=9, color=COLORS["ink"])
    ax.legend(fontsize=8); ax.tick_params(labelsize=7)
    fig2.tight_layout()
    b64_pc1 = fig_b64(fig2)

    return b64_res, b64_pc1

def plot_heatmap(name: str, df: pd.DataFrame) -> str | None:
    cols = ["Close", "ATR_res", "MFI_res", "STOCH_res", "PC1"]
    sub  = df[[c for c in cols if c in df.columns]].dropna()
    if len(sub) < 5:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    fig.patch.set_facecolor(COLORS["bg"]); ax.set_facecolor(COLORS["bg"])
    sns.heatmap(sub.corr(method="pearson"), annot=True, fmt=".2f",
                cmap="RdBu_r", vmin=-1, vmax=1, ax=ax,
                linewidths=0.5, annot_kws={"size": 9})
    ax.set_title(f"{name}  |  피어슨 상관 히트맵", fontsize=10,
                 fontweight="bold", color=COLORS["ink"])
    fig.tight_layout()
    return fig_b64(fig)

def plot_bar_metrics(all_results: dict, name: str) -> str:
    """성공률 / 상관계수 모델 비교 막대."""
    models = list(all_results.keys())
    accs   = [all_results[m].get("성공률(방향)", 0) if all_results[m] else 0 for m in models]
    corrs  = [all_results[m].get("상관계수", 0)     if all_results[m] else 0 for m in models]
    x      = np.arange(len(models))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle(f"{name}  |  모델별 성공률(뱡향) 및 상관 비교", fontsize=11,
                 fontweight="bold", color=COLORS["ink"])
    for ax in (a1, a2):
        ax.set_facecolor(COLORS["bg"])
        for sp in ax.spines.values(): sp.set_color(COLORS["line"])
    bar_c = [COLORS["red"], COLORS["green"], COLORS["gold"]][:len(models)]
    a1.bar(x, accs, color=bar_c, alpha=0.85, width=0.5)
    a1.set_xticks(x); a1.set_xticklabels(models, fontsize=8)
    a1.set_ylabel("성공률(방향)", fontsize=9); a1.set_ylim(0, 1)
    a1.axhline(0.5, color="#888", lw=0.8, linestyle="--")
    for i, v in enumerate(accs):
        a1.text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=8)
    a2.bar(x, corrs, color=bar_c, alpha=0.85, width=0.5)
    a2.set_xticks(x); a2.set_xticklabels(models, fontsize=8)
    a2.set_ylabel("상관계수 (Pearson r)", fontsize=9)
    for i, v in enumerate(corrs):
        a2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig_b64(fig)

# ═══════════════════════════════════════════════════════════════
#  8. 메인 파이프라인 (종목별)
# ═══════════════════════════════════════════════════════════════
def process_symbol(symbol: str, name: str):
    print(f"\n{'='*55}\n  [{name}]  {symbol}")
    df = fetch_weekly(symbol)
    df = build_indicators(symbol, df)
    df = calc_residuals(df)
    df, var_ratio, loadings = calc_pca(df)
    df = make_features(df)
    df = df[(df.index >= START_DATE) & (df.index <= END_DATE)].copy()
    n_valid = df["Target"].notna().sum()
    print(f"  분석 주봉: {len(df)}  (타깃 유효: {n_valid})")

    results, pred_data = {}, {}
    for mname, feats in MODEL_FEATS.items():
        metrics, pred_s, true_s = run_cv(df, feats)
        results[mname]   = metrics
        pred_data[mname] = (pred_s, true_s)
        if metrics:
            # 훈련/테스트 성공률을 나란히 출력 → 격차가 크면 과적합.
            tr_acc = metrics.get("훈련 성공률", np.nan)
            gap    = (tr_acc - metrics['성공률(방향)']) if pd.notna(tr_acc) else np.nan
            print(f"  {mname:16s}  R²={metrics['R²']:.3f}  "
                  f"상관={metrics['상관계수']:.3f}  "
                  f"성공률(훈련/테스트)={tr_acc:.1%}/{metrics['성공률(방향)']:.1%}  "
                  f"격차={gap:+.1%}")
        else:
            print(f"  {mname:16s}  ← 데이터 부족")

    return df, results, pred_data, var_ratio, loadings

# ═══════════════════════════════════════════════════════════════
#  9. HTML 생성
# ═══════════════════════════════════════════════════════════════

# 참고:
# 아래 HTML 생성부는 all_might.py를 직접 실행할 때 report.html을 만드는 보조 리포트 기능입니다.
# 현재 Streamlit 대시보드는 dashboard.py에서 all_might의 분석 결과와 시각화 함수를 불러와 사용합니다.
# 따라서 발표용 대시보드에는 직접 사용되지 않지만, 정적 리포트 백업 기능으로 보존합니다.
CSS = """
:root{--bg:#f9f6f0;--bg2:#efe9dd;--ink:#26201c;--soft:#5b524a;--line:#d8cfbe;
  --red:#9d3326;--green:#2f6f5e;--gold:#b07d27}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:'Segoe UI',system-ui,sans-serif;font-size:15px;line-height:1.65;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:0 24px}
.hero{padding:60px 0 36px;border-bottom:3px double var(--ink)}
.hero h1{font-size:clamp(28px,5vw,50px);margin:0 0 10px;font-weight:800;letter-spacing:-.02em}
.hero p{color:var(--soft);font-size:16px;margin:0 0 6px}
.hero .meta{margin-top:16px;font-size:12px;color:var(--soft);
  font-family:monospace;display:flex;gap:20px;flex-wrap:wrap}
.hero .meta b{color:var(--ink)}
.legend{display:flex;gap:18px;flex-wrap:wrap;margin:28px 0 0;font-size:13px}
.legend span{padding:5px 12px;border-radius:20px;font-weight:600}
.la{background:#fde8e5;color:var(--red)}
.lb{background:#dff0eb;color:var(--green)}
.lc{background:#fdf3e0;color:var(--gold)}

section.asset{margin:52px 0;padding:30px;background:#fbf8f1;
  border:1px solid var(--line);border-radius:14px}
section.asset h2{font-size:30px;font-weight:800;margin:0 0 4px;
  padding-bottom:10px;border-bottom:2px solid var(--ink)}
.asset-sub{font-size:12.5px;color:var(--soft);margin:4px 0 20px;font-family:monospace}

table.mt{border-collapse:collapse;width:100%;font-size:13px;
  margin:0 0 24px;font-family:monospace}
table.mt th{background:var(--ink);color:var(--bg);padding:8px 14px;
  text-align:right;font-weight:500;white-space:nowrap}
table.mt th:first-child{text-align:left}
table.mt td{padding:7px 14px;text-align:right;
  border-bottom:1px solid var(--line);white-space:nowrap}
table.mt td.mn{text-align:left;font-weight:700}
table.mt td.ma{color:var(--red)}
table.mt td.mb{color:var(--green)}
table.mt td.mc{color:var(--gold)}
table.mt tbody tr:hover{background:var(--bg2)}
.best{background:#fffde7;font-weight:700}

.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin:16px 0}
.g1{display:grid;grid-template-columns:1fr;gap:16px;margin:16px 0}
@media(max-width:700px){.g2{grid-template-columns:1fr}}
figure{margin:0}
figcaption{font-size:13px;font-family:monospace;color:var(--soft);
  font-weight:600;margin-bottom:4px}
img{width:100%;border-radius:9px;border:1px solid var(--line);display:block}

footer{margin:48px 0 52px;padding-top:20px;border-top:3px double var(--ink);
  font-size:13px;color:var(--soft)}
footer b{color:var(--ink)}
"""

# def metrics_table(results: dict) -> str: #**원본**
#     # '훈련 성공률'을 '성공률(방향)'(=테스트) 바로 앞에 배치해 둘을 나란히 비교.
#     keys   = ["R²", "RMSE(%)", "MAE(%)", "상관계수", "훈련 성공률", "성공률(방향)"]
#     mcls   = {"Model A (PC1)": "ma", "Model B (잔차3개)": "mb", "Model C (전체)": "mc"}
#     head   = "<th>모델</th>" + "".join(f"<th>{k}</th>" for k in keys)
#     rows   = ""

#     # 최고 성공률 찾기
#     best_acc = max(
#         (r["성공률(방향)"] for r in results.values() if r and "성공률(방향)" in r),
#         default=None,
#     )

#     for mname, r in results.items():
#         if r is None:
#             rows += f'<tr><td class="mn {mcls.get(mname,"")}">{mname}</td>' \
#                     + "<td colspan='6' style='color:#aaa'>데이터 부족</td></tr>"
#             continue
#         cells = ""
#         for k in keys:
#             v = r.get(k, np.nan)
#             if pd.isna(v):
#                 cells += "<td>-</td>"
#             elif k == "성공률(방향)":
#                 is_best = (best_acc is not None and abs(v - best_acc) < 1e-9)
#                 cls = ' class="best"' if is_best else ""
#                 cells += f"<td{cls}><b>{v:.1%}</b></td>"
#             elif k == "훈련 성공률":
#                 # 훈련 성공률은 비교용 → best 하이라이트 없이 흐린 글씨로 표시.
#                 cells += f"<td style='color:#9a9088'>{v:.1%}</td>"
#             elif k == "상관계수":
#                 cells += f"<td><b>{v:.4f}</b></td>"
#             elif k == "R²":
#                 cells += f"<td>{v:.4f}</td>"
#             else:
#                 cells += f"<td>{v:.2f}</td>"
#         rows += (f'<tr><td class="mn {mcls.get(mname,"")}">{mname}</td>'
#                  f'{cells}</tr>')

#     return (f'<table class="mt"><thead><tr>{head}</tr></thead>'
#             f'<tbody>{rows}</tbody></table>')
def metrics_table(results: dict) -> str:
    """
    모델별 성능표를 HTML 카드형 표로 반환한다.
    Streamlit에서는 st.components.v1.html()로 렌더링한다.
    """

    keys = ["R²", "RMSE(%)", "MAE(%)", "상관계수", "훈련 성공률", "성공률(방향)"]

    model_labels = {
        "Model A (PC1)": "Model A<br><span>PC1 통합</span>",
        "Model B (잔차3개)": "Model B<br><span>Residual 3개</span>",
        "Model C (전체)": "Model C<br><span>PC1 + Residual</span>",
    }

    style = """
    <style>
        .metric-card {
            background: #ffffff;
            border: 1px solid #e3dacb;
            border-radius: 14px;
            padding: 14px 16px 16px 16px;
            margin: 8px 0 18px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.04);
            font-family: 'Segoe UI', system-ui, sans-serif;
        }

        .metric-title {
            font-size: 16px;
            font-weight: 700;
            color: #2f2a25;
            margin-bottom: 10px;
        }

        table.metric-table {
            border-collapse: separate;
            border-spacing: 0;
            width: 100%;
            overflow: hidden;
            border-radius: 10px;
            border: 1px solid #e7dfd2;
            font-size: 16px;
        }

        table.metric-table th {
            background: #f3eee5;
            color: #2f2a25;
            padding: 10px 12px;
            text-align: center;
            font-weight: 700;
            border-bottom: 1px solid #e0d6c8;
            white-space: nowrap;
        }

        table.metric-table td {
            padding: 11px 12px;
            text-align: center;
            border-bottom: 1px solid #eee7dd;
            color: #332c26;
            white-space: nowrap;
        }

        table.metric-table tr:last-child td {
            border-bottom: none;
        }

        table.metric-table tbody tr:hover {
            background: #fbf8f1;
        }

        td.model-name {
            font-weight: 700;
            text-align: center;
            background: #fcfaf6;
            color: #26201c;
            min-width: 120px;
        }

        td.model-name span {
            font-size: 13px;
            font-weight: 500;
            color: #7a7067;
        }

        .pos {
            background: #eaf5ef;
            color: #1f5f4b;
            font-weight: 700;
        }

        .neg {
            background: #fbecea;
            color: #8a2d24;
            font-weight: 700;
        }

        .neutral {
            background: #f7f4ee;
            color: #5b524a;
        }

        .best {
            background: #fff6d8;
            color: #7a5600;
            font-weight: 800;
        }

        .train {
            background: #f7f4ee;
            color: #8b8178;
            font-weight: 600;
        }

        .metric-note {
            margin-top: 10px;
            font-size: 16px;
            line-height: 1.55;
            color: #6a6058;
        }

        .metric-note b {
            color: #2f2a25;
        }
    </style>
    """

    best_acc = max(
        (r["성공률(방향)"] for r in results.values() if r and "성공률(방향)" in r),
        default=None,
    )

    head = "<th>모델</th>" + "".join(f"<th>{k}</th>" for k in keys)
    rows = ""

    for mname, r in results.items():
        display_name = model_labels.get(mname, mname)

        if r is None:
            rows += (
                f"<tr><td class='model-name'>{display_name}</td>"
                "<td colspan='6' class='neutral'>데이터 부족</td></tr>"
            )
            continue

        cells = ""

        for k in keys:
            v = r.get(k, np.nan)

            if pd.isna(v):
                cells += "<td class='neutral'>-</td>"

            elif k == "R²":
                cls = "pos" if v >= 0 else "neg"
                cells += f"<td class='{cls}'>{v:.4f}</td>"

            elif k == "상관계수":
                cls = "pos" if v >= 0 else "neg"
                cells += f"<td class='{cls}'>{v:.4f}</td>"

            elif k == "성공률(방향)":
                is_best = (best_acc is not None and abs(v - best_acc) < 1e-9)
                if is_best:
                    cls = "best"
                elif v >= 0.50:
                    cls = "pos"
                else:
                    cls = "neg"
                cells += f"<td class='{cls}'>{v:.1%}</td>"

            elif k == "훈련 성공률":
                cells += f"<td class='train'>{v:.1%}</td>"

            elif k in ["RMSE(%)", "MAE(%)"]:
                cells += f"<td>{v:.2f}</td>"

            else:
                cells += f"<td>{v}</td>"

        rows += f"<tr><td class='model-name'>{display_name}</td>{cells}</tr>"

    note = """
    <div class="metric-note">
        <b>훈련 성공률</b>은 마지막 학습 구간에서 모델이 방향성을 맞힌 비율이며,
        <b>성공률(방향)</b>은 테스트 구간에서 다음 주 수익률의 상승·하락 방향을 맞힌 비율입니다.
        두 값의 차이가 크면 과적합 가능성을 함께 점검합니다.
    </div>
    """

    return f"""
    {style}
    <div class="metric-card">
        <div class="metric-title">모델별 성능 비교</div>
        <table class="metric-table">
            <thead><tr>{head}</tr></thead>
            <tbody>{rows}</tbody>
        </table>
        {note}
    </div>
    """



def build_asset_section(name: str, data_tuple) -> str:
    df, results, pred_data, var_ratio, loadings = data_tuple

    # 차트 생성
    b64_res, b64_pc1 = plot_residuals_and_pc1(name, df, var_ratio, loadings)
    b64_hm           = plot_heatmap(name, df)
    b64_bar          = plot_bar_metrics(results, name)

    pred_figs = ""
    for mname, (pred_s, true_s) in pred_data.items():
        if pred_s is not None and results.get(mname):
            b64 = plot_pred(name, mname, true_s, pred_s, results[mname])
            pred_figs += (f'<figure><figcaption>{mname}</figcaption>'
                          f'<img src="data:image/png;base64,{b64}"></figure>')

    hm_html = ""
    if b64_hm:
        hm_html = (f'<div class="g2">'
                   f'<figure><figcaption>피어슨 상관 히트맵</figcaption>'
                   f'<img src="data:image/png;base64,{b64_hm}"></figure>'
                   f'<figure><figcaption>모델 성능 비교</figcaption>'
                   f'<img src="data:image/png;base64,{b64_bar}"></figure></div>')
    else:
        hm_html = (f'<div class="g1">'
                   f'<figure><figcaption>모델 성능 비교</figcaption>'
                   f'<img src="data:image/png;base64,{b64_bar}"></figure></div>')

    return f"""<section class="asset" id="{name}">
  <h2>{name}</h2>
  <div class="asset-sub">PC1 설명분산: <b>{var_ratio:.1%}</b> &nbsp;|&nbsp;
    loadings: ATR {loadings[0]:+.2f} · MFI {loadings[1]:+.2f} · STOCH {loadings[2]:+.2f}
    &nbsp;|&nbsp; TimeSeriesSplit n={N_SPLITS} · 타깃: t+1 주봉 수익률
  </div>
  {metrics_table(results)}
  {hm_html}
  <div class="g2">
    <figure><figcaption>OLS 잔차 (심리 성분)</figcaption>
      <img src="data:image/png;base64,{b64_res}"></figure>
    <figure><figcaption>투자자 심리지수 PC1</figcaption>
      <img src="data:image/png;base64,{b64_pc1}"></figure>
  </div>
  <div class="g1">{pred_figs}</div>
</section>"""

def build_html(all_data: dict) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    sections = "\n".join(build_asset_section(n, d) for n, d in all_data.items())
    nav = " · ".join(f'<a href="#{n}" style="color:var(--green)">{n}</a>'
                     for n in all_data)

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>XGBoost 심리지수 주가 예측 리포트</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<header class="hero">
  <h1>XGBoost 심리지수 주가 예측</h1>
  <p>비트코인 · 삼성전자 · 코스닥 &nbsp;|&nbsp; 최근 5년 주봉 &nbsp;|&nbsp;
     OLS 잔차 → PCA(투자자 심리지수) → XGBoost 3모델 비교</p>
  <div class="meta">
    <span>생성: <b>{now}</b></span>
    <span>데이터: <b>yfinance 주봉</b></span>
    <span>검증: <b>TimeSeriesSplit(n={N_SPLITS})</b></span>
    <span>성공률 = 다음 주 방향(상승/하락) 예측 정확도</span>
  </div>
  <div class="legend">
    <span class="la">Model A &nbsp; Close + PC1(통합 심리지수)</span>
    <span class="lb">Model B &nbsp; Close + 잔차3개(ATR·MFI·STOCH)</span>
    <span class="lc">Model C &nbsp; Close + PC1 + 잔차3개 (혼합)</span>
  </div>
  <p style="margin-top:18px;font-size:13px;color:var(--soft)">
    바로가기: {nav}
  </p>
</header>

{sections}

<footer>
  <p><b>파이프라인 ·</b>
    yfinance 주봉 수집 → MFI/ATR/Stochastic 산출 →
    OLS(수익률·모멘텀·변동성 통제) 잔차 추출 →
    StandardScaler + PCA → PC1(투자자 심리지수) →
    XGBoost 회귀 (TimeSeriesSplit 교차검증)</p>
  <p><b>평가 지표 ·</b>
    R²(설명력) · RMSE/MAE(수익률 오차, %) · 상관계수(Pearson r) ·
    <b>성공률(다음 주 상승/하락 방향 예측 정확도)</b> —
    타겟: 절대 종가 → <b>t+1 주봉 수익률</b>로 변경.
    PCA는 각 CV 폴드 훈련 데이터에서만 피팅(미래 누출 방지).</p>
  <p><b>과적합 점검 ·</b>
    <b>훈련 성공률</b>(마지막 폴드 훈련 구간)과 <b>성공률(테스트)</b>을 비교 —
    훈련이 테스트보다 크게 높으면 과적합 신호(트리 수·max_depth 축소 검토).
    훈련 ≈ 테스트면 건강한 학습.</p>
  <p><b>노트 ·</b>
    코스닥은 발행주식수가 없어 표준 ATR(Wilder RMA) 사용.
    MFI는 거래량이 0인 기간 NaN 처리.
    주가 예측 모델은 참고용이며 투자 결정 근거로 사용하지 마십시오.</p>
</footer>
</div></body></html>"""

# ═══════════════════════════════════════════════════════════════
#  10. 실행
# ═══════════════════════════════════════════════════════════════
def main():
    print("XGBoost 심리지수 주가 예측 시작")
    print(f"기간: {START_DATE.date()} ~ {END_DATE.date()}  (5년 주봉)\n")

    all_data = {}
    for symbol, name in SYMBOLS.items():
        try:
            all_data[name] = process_symbol(symbol, name)
        except Exception as e:
            print(f"[{name}] 처리 실패: {e}")
            import traceback; traceback.print_exc()

    if not all_data:
        print("처리된 종목이 없습니다. 네트워크/설치 확인 필요.")
        return

    html = build_html(all_data)
    out  = "report.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n{'='*55}")
    print(f"✔ 리포트 저장 완료: {out}")
    try:
        import os
        webbrowser.open("file://" + os.path.realpath(out))
    except Exception:
        pass

if __name__ == "__main__":
    main()
