from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


TICKER_MAP = {
    "삼성전자": "005930.KS",
    "코스피": "^KS11",
    "비트코인": "BTC-USD",
}


def calc_mfi(df: pd.DataFrame, period: int = 10) -> pd.Series:
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]
    price_diff = typical_price.diff()

    positive_flow = pd.Series(
        np.where(price_diff > 0, raw_money_flow, 0),
        index=df.index,
    )
    negative_flow = pd.Series(
        np.where(price_diff < 0, raw_money_flow, 0),
        index=df.index,
    )

    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        money_flow_ratio = positive_sum / negative_sum
        mfi = np.where(
            negative_sum == 0,
            100,
            100 - (100 / (1 + money_flow_ratio)),
        )

    return pd.Series(mfi, index=df.index, name=f"MFI_{period}")


def calc_atr(df: pd.DataFrame, period: int = 10) -> pd.Series:
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().rename(f"ATR_{period}")


def calc_stoch(df: pd.DataFrame, period: int = 10, smooth: int = 3) -> pd.Series:
    lowest_low = df["Low"].rolling(window=period).min()
    highest_high = df["High"].rolling(window=period).max()
    fast_k = 100 * (df["Close"] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    return fast_k.rolling(window=smooth).mean().rename(f"STOCHk_{period}_{smooth}_3")


def get_recent_data(asset_name: str) -> pd.DataFrame:
    ticker = TICKER_MAP.get(asset_name)
    if ticker is None:
        supported = ", ".join(TICKER_MAP)
        raise ValueError(f"지원하지 않는 자산입니다: {asset_name}. 지원 목록: {supported}")

    start_date = datetime.now() - timedelta(days=900)
    df = yf.download(ticker, start=start_date, interval="1wk", progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna(subset=["Close"]).copy()
    if "Volume" not in df.columns or df["Volume"].isna().all():
        df["Volume"] = 1

    df["MFI_10"] = calc_mfi(df, 10)
    df["ATR_10"] = calc_atr(df, 10)
    df["STOCHk_10_3_3"] = calc_stoch(df, 10, 3)

    return df.dropna(subset=["Close", "MFI_10", "ATR_10", "STOCHk_10_3_3"]).copy()
