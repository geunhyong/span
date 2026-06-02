import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import Samsung_crawling as sc

# baseline 실험은 train_pipeline.py 의 작업구성을 맞추어 진행합니다.

# Samsung_crawling.py
# train_pipeline.py와 같은 방식으로 데이터 구성
# 같은 Target_Log_Return 사용
# 같은 80:20 시계열 split
# 같은 XGBRegressor 파라미터 사용
# Naive / Price-only baseline 결과 출력
import os
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 보성님 train_pipeline.py와 동일한 전처리 기준
import Samsung_crawling as sc


def calculate_da(y_true, y_pred):
    """
    방향 정확도 Directional Accuracy 계산.
    train_pipeline.py의 calculate_da와 개념을 맞추되,
    결과는 0~100 퍼센트 단위로 반환한다.
    """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)

    match = (true_sign == pred_sign) & (true_sign != 0)

    if len(y_true) == 0:
        return np.nan

    return np.sum(match) / len(y_true) * 100


def evaluate_baseline(y_true, y_pred):
    """
    baseline 모델 성능 평가.
    train_pipeline.py의 결과표와 맞추기 위해 R2, RMSE, MAE, DA를 반환한다.
    """
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "DA": calculate_da(y_true, y_pred),
    }


def build_panel_dataset():
    """
    Samsung_crawling.py 기준으로 3개 자산 데이터를 수집하고
    train_pipeline.py와 같은 패널 데이터 구조를 만든다.
    """
    panel_list = []

    for ticker, name in sc.SYMBOLS.items():
        print(f"[{name}] 데이터 수집 및 전처리 시작")

        df_asset = sc.build_weekly_df(
            symbol=ticker,
            label=name,
            start_date=sc.START_DATE,
            end_date=sc.END_DATE,
            warmup_start=sc.WARMUP_START,
        )

        df_asset["Ticker"] = name
        panel_list.append(df_asset)

    df_full = pd.concat(panel_list)
    return df_full


def make_price_only_dataset(df_full):
    """
    Price-only baseline용 데이터셋 구성.

    사용 feature:
    - Log_Return_lag1 ~ Log_Return_lag5
    - Asset 더미 변수

    Target:
    - Target_Log_Return
    """
    df_processed = df_full.copy()

    # train_pipeline.py와 동일하게 로그 수익률 사용
    df_processed["Log_Return"] = np.log(
        df_processed["Close"] / df_processed.groupby("Ticker")["Close"].shift(1)
    )

    # 다음 주 로그 수익률
    df_processed["Target_Log_Return"] = (
        df_processed.groupby("Ticker")["Log_Return"].shift(-1)
    )

    # 과거 1~5주 로그 수익률 lag
    for i in range(1, 6):
        df_processed[f"Log_Return_lag{i}"] = (
            df_processed.groupby("Ticker")["Log_Return"].shift(i)
        )

    # 자산 더미 변수
    df_processed = pd.get_dummies(
        df_processed,
        columns=["Ticker"],
        prefix="Asset",
        dtype=int,
    )

    # 결측치 제거
    df_final = df_processed.dropna().copy()
    df_final = df_final.sort_index()

    return df_final


def run_baseline_experiment():
    print("=" * 60)
    print("📊 Baseline 실험 시작")
    print("=" * 60)

    df_full = build_panel_dataset()
    df_final = make_price_only_dataset(df_full)

    y = df_final["Target_Log_Return"]

    price_lags = [
        col for col in df_final.columns
        if "Log_Return_lag" in col or col.startswith("Asset_")
    ]

    X_price = df_final[price_lags]

    # train_pipeline.py와 동일하게 80:20 시계열 분할
    split_idx = int(len(df_final) * 0.8)

    X_train = X_price.iloc[:split_idx]
    X_test = X_price.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    results = []

    # ---------------------------------------------------------
    # Baseline 1: Naive Baseline
    # 예측 로그 수익률을 0으로 둔다.
    # ---------------------------------------------------------
    y_pred_naive = np.zeros_like(y_test)

    naive_metrics = evaluate_baseline(y_test, y_pred_naive)
    results.append({
        "Model": "Naive Baseline",
        **naive_metrics,
    })

    # ---------------------------------------------------------
    # Baseline 2: Price-only XGBoost
    # 가격 lag + Asset 더미만 사용한다.
    # ---------------------------------------------------------
    model_price = XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        min_child_weight=5,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
    )

    model_price.fit(X_train, y_train, verbose=False)
    y_pred_price = model_price.predict(X_test)

    price_metrics = evaluate_baseline(y_test, y_pred_price)
    results.append({
        "Model": "Price-only XGBoost",
        **price_metrics,
    })

    result_df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("📊 Baseline 실험 결과")
    print("=" * 60)

    pd.set_option("display.float_format", "{:.4f}".format)
    print(result_df.to_string(index=False))

    print("\n사용 feature:")
    print(list(X_price.columns))

    print("\nTrain/Test 샘플 수:")
    print(f"Train: {len(X_train)}")
    print(f"Test : {len(X_test)}")

    return result_df, X_price, y_test
# TODO:
# baseline 실험 결과를 dashboard.py에서 직접 표시할 수 있도록
# 아래 흐름으로 개선 예정
#
# 1. models_baseline.py 실행
# 2. baseline_results.csv 저장
# 3. dashboard.py에서 baseline_results.csv를 읽어 Baseline 성능표로 표시
# TODO:
# 추후 아래 result_df를 csv로 저장하여 dashboard.py에서 불러오도록 연결
# result_df.to_csv("baseline_results.csv", index=False, encoding="utf-8-sig")
###  `baseline_results.csv`로 저장한 뒤, `dashboard.py`에서 해당 csv를 읽어 Baseline 성능표로 표시하도록 연결한다.

# Naive Baseline
# Price-only XGBoost
# Model A
# Model B
# Model C

# 그리고 비교 기준은:

# R²
# RMSE
# MAE
# 방향성 성공률
# Price-only 대비 방향성 개선폭
# 해석 메모
if __name__ == "__main__":
    result_df, X_price, y_test = run_baseline_experiment()