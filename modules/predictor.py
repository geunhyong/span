# modules/predictor.py
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression

class QuantPredictor:
    def __init__(self):
        # 1. 3개의 모델 및 전처리 객체 모두 로드
        self.model_A = joblib.load('models/best_xgboost_panel_model_A.pkl')
        self.model_B = joblib.load('models/best_xgboost_panel_model_B.pkl')
        self.model_C = joblib.load('models/best_xgboost_panel_model_C.pkl')
        self.scaler = joblib.load('models/scaler.pkl')
        self.pca = joblib.load('models/pca.pkl')

    def _extract_residuals(self, df):
        """ 실시간 데이터에 대한 OLS 잔차 추출 (train_pipeline과 동일 로직) """
        df_res = df.copy()
        df_res['RET'] = df_res['Close'].pct_change()
        df_res['MOM'] = df_res['RET'].shift(1).rolling(window=10).sum()
        df_res['VOL'] = df_res['RET'].rolling(window=11).var()
        
        control_features = ['RET', 'MOM', 'VOL']
        target_indicators = ['ATR_10', 'MFI_10', 'STOCHk_10_3_3']
        
        df_clean = df_res.dropna(subset=control_features + target_indicators).copy()
        X_control = df_clean[control_features]
        
        for indicator in target_indicators:
            lr = LinearRegression()
            lr.fit(X_control, df_clean[indicator])
            df_clean[f'{indicator}_res'] = df_clean[indicator] - lr.predict(X_control)
            
        return df_clean

    def get_prediction(self, asset_name, df_live, model_type="B"):
        """ 선택된 모델(A/B/C)에 맞춰 실시간 추론 수행 """
        # 1. 잔차 추출
        df_live = self._extract_residuals(df_live)

        # 2. PCA 심리지수 산출
        residual_cols = ['ATR_10_res', 'MFI_10_res', 'STOCHk_10_3_3_res']
        X_res = df_live[residual_cols]
        X_scaled = self.scaler.transform(X_res) # 이미 계산된 스케일링 값
        sentiment_score = self.pca.transform(X_scaled)

        # 부호 보정
        mfi_idx = residual_cols.index('MFI_10_res')
        if self.pca.components_[0][mfi_idx] < 0:
            sentiment_score = -sentiment_score
        df_live['Investor_Sentiment_PC1'] = sentiment_score

        # 3. 로그 수익률 및 시차 변수 생성
        df_live['Log_Return'] = np.log(df_live['Close'] / df_live['Close'].shift(1))
        for i in range(1, 6):
            df_live[f'Log_Return_lag{i}'] = df_live['Log_Return'].shift(i)

        df_live = df_live.dropna().copy()
        
        # 내일 예측을 위한 가장 최근(마지막) 행 추출
        current_data = df_live.iloc[-1:].copy()

        # 4. 공통 피처 (가격 Lags + Asset 원-핫 인코딩)
        assets = ['Asset_비트코인', 'Asset_삼성전자', 'Asset_코스피']
        for a in assets:
            current_data[a] = 1 if a == f'Asset_{asset_name}' else 0

        # 5. 선택된 모델에 따른 피처 조합 및 예측
        if model_type == "A":
            model = self.model_A
        elif model_type == "B":
            model = self.model_B
        else:
            model = self.model_C

        # 모델이 학습 시 사용했던 컬럼 목록(feature_names_in_)을 그대로 가져옴
        expected_cols = model.feature_names_in_
        
        # 방어 코드: 모델은 요구하지만 현재 데이터에 없는 컬럼을 0으로 채움
        for col in expected_cols:
            if col not in current_data.columns:
                current_data[col] = 0
                
        X_pred = current_data[expected_cols]

        # 예측 수행
        pred_log_return = model.predict(X_pred)[0]
        direction = "UP" if pred_log_return > 0 else "DOWN"

        return {
            "pred_log_return": pred_log_return,
            "direction": direction,
            "df_plot": df_live,
            "current_data": current_data.iloc[0].to_dict(),
            # 💡 [추가] 가장 최근(마지막 행)의 스케일링된 잔차 값 3개 전달
            "scaled_residuals": X_scaled[-1].tolist() 
        }