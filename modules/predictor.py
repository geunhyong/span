from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_NAMES = ["비트코인", "삼성전자", "코스피"]


class QuantPredictor:
    def __init__(self):
        self.model_A = joblib.load(MODEL_DIR / "best_xgboost_panel_model_A.pkl")
        self.model_B = joblib.load(MODEL_DIR / "best_xgboost_panel_model_B.pkl")
        self.model_C = joblib.load(MODEL_DIR / "best_xgboost_panel_model_C.pkl")
        self.scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        self.pca = joblib.load(MODEL_DIR / "pca.pkl")

    def _extract_residuals(self, df):
        result = df.copy()
        result["RET"] = result["Close"].pct_change()
        result["MOM"] = result["RET"].shift(1).rolling(window=10).sum()
        result["VOL"] = result["RET"].rolling(window=11).var()

        control_features = ["RET", "MOM", "VOL"]
        target_indicators = ["ATR_10", "MFI_10", "STOCHk_10_3_3"]

        clean = result.dropna(subset=control_features + target_indicators).copy()
        if clean.empty:
            raise ValueError("잔차 계산에 필요한 유효 데이터가 부족합니다. 더 긴 기간의 주봉 데이터가 필요합니다.")

        controls = clean[control_features]
        for indicator in target_indicators:
            model = LinearRegression()
            model.fit(controls, clean[indicator])
            clean[f"{indicator}_res"] = clean[indicator] - model.predict(controls)

        return clean

    def _select_model(self, model_type: str):
        if model_type == "A":
            return self.model_A
        if model_type == "B":
            return self.model_B
        if model_type == "C":
            return self.model_C
        raise ValueError(f"지원하지 않는 모델 타입입니다: {model_type}")

    def get_prediction(self, asset_name, df_live, model_type="B"):
        df_live = self._extract_residuals(df_live)

        residual_cols = ["ATR_10_res", "MFI_10_res", "STOCHk_10_3_3_res"]
        scaled_residuals = self.scaler.transform(df_live[residual_cols])
        sentiment_score = self.pca.transform(scaled_residuals)

        mfi_idx = residual_cols.index("MFI_10_res")
        if self.pca.components_[0][mfi_idx] < 0:
            sentiment_score = -sentiment_score
        df_live["Investor_Sentiment_PC1"] = sentiment_score

        df_live["Log_Return"] = np.log(df_live["Close"] / df_live["Close"].shift(1))
        for lag in range(1, 6):
            df_live[f"Log_Return_lag{lag}"] = df_live["Log_Return"].shift(lag)

        df_live = df_live.dropna().copy()
        if df_live.empty:
            raise ValueError("예측 피처 생성 후 남은 유효 데이터가 없습니다. 더 긴 기간의 주봉 데이터가 필요합니다.")

        current_data = df_live.iloc[-1:].copy()

        for name in ASSET_NAMES:
            current_data[f"Asset_{name}"] = 1 if name == asset_name else 0

        model = self._select_model(model_type)
        expected_cols = model.feature_names_in_

        for col in expected_cols:
            if col not in current_data.columns:
                current_data[col] = 0

        prediction_input = current_data[expected_cols]
        pred_log_return = float(model.predict(prediction_input)[0])
        direction = "UP" if pred_log_return > 0 else "DOWN"

        return {
            "pred_log_return": pred_log_return,
            "direction": direction,
            "df_plot": df_live,
            "current_data": current_data.iloc[0].to_dict(),
            "scaled_residuals": scaled_residuals[-1].tolist(),
        }
