import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

# 조원분의 크롤링 모듈 임포트
import Samsung_crawling as sc

warnings.filterwarnings('ignore')

def extract_sentiment_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """ [회의록 반영] 제어 변수(RET, MOM, VOL)를 통한 OLS 잔차 추출 """
    df_res = df.copy()
    
    # 1. 제어 변수(X_control) 생성 (10주 기준 반영)
    df_res['RET'] = df_res['Close'].pct_change()
    df_res['MOM'] = df_res['RET'].shift(1).rolling(window=10).sum()
    df_res['VOL'] = df_res['RET'].rolling(window=11).var()
    
    control_features = ['RET', 'MOM', 'VOL']
    target_indicators = ['ATR_10', 'MFI_10', 'STOCHk_10_3_3']
    
    # 결측치 제거 후 OLS 회귀 수행
    df_clean = df_res.dropna(subset=control_features + target_indicators).copy()
    X_control = df_clean[control_features]
    
    # 2. 각 지표별 선형 회귀 후 잔차(Residual) 산출
    for indicator in target_indicators:
        lr = LinearRegression()
        lr.fit(X_control, df_clean[indicator])
        df_clean[f'{indicator}_res'] = df_clean[indicator] - lr.predict(X_control)
        
    return df_clean

def create_sentiment_index(df: pd.DataFrame) -> pd.DataFrame:
    """ PCA 기반 투자자 심리지수 산출 및 객체 저장 """
    residual_cols = ['ATR_10_res', 'MFI_10_res', 'STOCHk_10_3_3_res']
    X_res = df[residual_cols]
    
    # 1. 스케일러 학습
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_res)
    
    # 2. PCA 학습
    pca = PCA(n_components=1)
    sentiment_score = pca.fit_transform(X_scaled)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(pca, 'models/pca.pkl')
    
    # MFI 잔차를 기준으로 양수(+)가 긍정적 심리가 되도록 부호 보정
    mfi_idx = residual_cols.index('MFI_10_res')
    if pca.components_[0][mfi_idx] < 0:
        sentiment_score = -sentiment_score
        
    df['Investor_Sentiment_PC1'] = sentiment_score
    return df

def calculate_da(y_true, y_pred):
    """ 방향 정확도 (Directional Accuracy) 계산 """
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    match = (true_sign == pred_sign) & (true_sign != 0)
    return np.sum(match) / len(y_true) * 100

def train_and_evaluate(model_name, X_train, y_train, X_test, y_test):
    """ XGBoost 모델 학습 및 평가 헬퍼 함수 """
    model = XGBRegressor(
        n_estimators=500, learning_rate=0.01, max_depth=3, min_child_weight=5,
        reg_alpha=0.5, reg_lambda=1.0, random_state=42
    )
    model.fit(X_train, y_train, verbose=False)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    da = calculate_da(y_test, y_pred)
    
    return model, r2, rmse, da, y_pred

def main():
    print("=========================================================")
    print(" 📡 [1단계] 패널 데이터 로드 및 잔차 추출")
    print("=========================================================")
    panel_list = []
    for ticker, name in sc.SYMBOLS.items():
        df_asset = sc.build_weekly_df(
            symbol=ticker, label=name, 
            start_date=sc.START_DATE, end_date=sc.END_DATE, warmup_start=sc.WARMUP_START
        )
        df_asset['Ticker'] = name
        panel_list.append(df_asset)
        
    df_full = pd.concat(panel_list)
    
    # 💡 1. OLS 잔차 추출 로직 적용
    df_processed = extract_sentiment_residuals(df_full)
    
    print("=========================================================")
    print(" 🧠 [2단계] 심리지수(PCA) 추출")
    print("=========================================================")
    # 💡 2. 추출된 잔차를 바탕으로 PCA 심리지수 생성
    df_processed = create_sentiment_index(df_processed)

    print("=========================================================")
    print(" ⚙️ [3단계] Feature Engineering (로그 수익률 및 시차 변수)")
    print("=========================================================")
    # 로그 수익률 계산 (절대 가격 예측의 함정 회피)
    df_processed['Log_Return'] = np.log(df_processed['Close'] / df_processed.groupby('Ticker')['Close'].shift(1))
    df_processed['Target_Log_Return'] = df_processed.groupby('Ticker')['Log_Return'].shift(-1)
    
    # 기본 가격 시차 변수 (Lag 1~5)
    for i in range(1, 6):
        df_processed[f'Log_Return_lag{i}'] = df_processed.groupby('Ticker')['Log_Return'].shift(i)
        
    # 원-핫 인코딩
    df_processed = pd.get_dummies(df_processed, columns=['Ticker'], prefix='Asset', dtype=int)
    
    # 결측치 최종 제거
    df_final = df_processed.dropna().copy()
    df_final = df_final.sort_index()
    
    print("=========================================================")
    print(" 🤖 [4단계] Model A/B/C 및 Baseline 대조 실험")
    print("=========================================================")
    
    # 타겟 분리
    y = df_final['Target_Log_Return']
    
    # 💡 피처 그룹 정의 (회의록 4장 반영)
    price_lags = [col for col in df_final.columns if 'Log_Return_lag' in col or col.startswith('Asset_')]
    residual_cols = ['ATR_10_res', 'MFI_10_res', 'STOCHk_10_3_3_res']
    pca_col = ['Investor_Sentiment_PC1']
    
    # 데이터 분할 (Train 80%, Test 20%)
    split_idx = int(len(df_final) * 0.8)
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    results = []

    # ---------------------------------------------------------
    # Baseline 1: Naive Baseline (t-1 종가 유지 = 예측값 0)
    # ---------------------------------------------------------
    y_pred_naive = np.zeros_like(y_test)
    da_naive = calculate_da(y_test, y_pred_naive) # 부호가 0이므로 매칭 안됨
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    results.append({"Model": "Naive Baseline", "R2": 0.0, "RMSE": rmse_naive, "DA": 50.0})

    # ---------------------------------------------------------
    # Baseline 2: Price-only XGBoost
    # ---------------------------------------------------------
    X_base = df_final[price_lags]
    X_train_b, X_test_b = X_base.iloc[:split_idx], X_base.iloc[split_idx:]
    _, r2_b, rmse_b, da_b, _ = train_and_evaluate("Price-only", X_train_b, y_train, X_test_b, y_test)
    results.append({"Model": "Price-only Baseline", "R2": r2_b, "RMSE": rmse_b, "DA": da_b})

    # ---------------------------------------------------------
    # Model A: 통합 심리형 (Price + PC1)
    # ---------------------------------------------------------
    X_A = df_final[price_lags + pca_col]
    X_train_A, X_test_A = X_A.iloc[:split_idx], X_A.iloc[split_idx:]
    model_A, r2_A, rmse_A, da_A, _ = train_and_evaluate("Model A", X_train_A, y_train, X_test_A, y_test)
    results.append({"Model": "Model A (PCA 통합)", "R2": r2_A, "RMSE": rmse_A, "DA": da_A})
    joblib.dump(model_A, 'models/best_xgboost_panel_model_A.pkl') # 메인 모델 저장

    # ---------------------------------------------------------
    # Model B: 세부 심리형 (Price + 개별 잔차 3개)
    # ---------------------------------------------------------
    X_B = df_final[price_lags + residual_cols]
    X_train_B, X_test_B = X_B.iloc[:split_idx], X_B.iloc[split_idx:]
    model_B, r2_B, rmse_B, da_B, _ = train_and_evaluate("Model B", X_train_B, y_train, X_test_B, y_test)
    results.append({"Model": "Model B (세부 잔차)", "R2": r2_B, "RMSE": rmse_B, "DA": da_B})
    joblib.dump(model_B, 'models/best_xgboost_panel_model_B.pkl') # 모델 B 저장

    # ---------------------------------------------------------
    # Model C: 혼용 탐색형 (Price + PC1 + 개별 잔차)
    # ---------------------------------------------------------
    X_C = df_final[price_lags + pca_col + residual_cols]
    X_train_C, X_test_C = X_C.iloc[:split_idx], X_C.iloc[split_idx:]
    model_C, r2_C, rmse_C, da_C, _ = train_and_evaluate("Model C", X_train_C, y_train, X_test_C, y_test)
    results.append({"Model": "Model C (전체 혼용)", "R2": r2_C, "RMSE": rmse_C, "DA": da_C})
    joblib.dump(model_C, 'models/best_xgboost_panel_model_C.pkl') # 모델 C 저장
    print("\n=========================================================")
    print(" 📊 [최종] Model 대조 실험 결과 요약 (Test Set)")
    print("=========================================================")
    res_df = pd.DataFrame(results)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(res_df.to_string(index=False))
    
    print("\n💾 훈련된 Model A, B, C 저장 완료: 'models/best_xgboost_panel_model_A.pkl', 'models/best_xgboost_panel_model_B.pkl', 'models/best_xgboost_panel_model_C.pkl'")

if __name__ == '__main__':
    main()