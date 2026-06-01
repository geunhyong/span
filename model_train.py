import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.stattools import adfuller
import joblib
import warnings

# 불필요한 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# =========================================================
# 1. 더미 데이터 생성 (실제 데이터 수집 전 테스트용)
# =========================================================
np.random.seed(42)
n_samples = 500

# 의도적으로 단위근이 있는 '비정상 시계열' 생성
close_price = np.cumsum(np.random.normal(0, 1, n_samples)) + 100
investor_sentiment = np.cumsum(np.random.normal(0, 0.5, n_samples))

X_train = pd.DataFrame({
    'Close': close_price,
    'Investor_Sentiment': investor_sentiment
})

print("📊 [원본 데이터 처음 5행]")
print(X_train.head(), "\n")

# =========================================================
# 2. ADF 검정 및 조건부 차분 함수
# =========================================================
def check_stationarity_and_diff(df):
    print("🔍 [ADF 단위근 검정 시작]")
    is_stationary = True
    
    for col in df.columns:
        result = adfuller(df[col])
        p_value = result[1]
        print(f" - {col} p-value: {p_value:.4f}")
        
        if p_value >= 0.05:
            print(f"   -> ⚠️ 비정상성 확인 (차분 필요)")
            is_stationary = False
        else:
            print(f"   -> ✅ 정상성 확인")
            
    if not is_stationary:
        print(f"\n🚨 [조치] 불안정한 변수가 발견되어 데이터프레임 전체에 1차 차분을 수행합니다.")
        df_diff = df.diff().dropna() 
        
        print("\n🔍 [1차 차분 후 재검정]")
        for col in df_diff.columns:
            result_diff = adfuller(df_diff[col])
            print(f" - {col} 차분 후 p-value: {result_diff[1]:.4f}")
        return df_diff
    else:
        print("\n✅ 모든 변수가 정상성을 만족합니다. 차분 없이 원본을 사용합니다.")
        return df

# 파이프라인 실행
X_train_diff = check_stationarity_and_diff(X_train)

# =========================================================
# 3. 모델 학습을 위한 Feature Engineering (시차 변수 생성)
# =========================================================
X_data = X_train_diff.copy()

# 과거 데이터를 Feature로 사용 (어제의 주가, 어제의 심리지수)
X_data['Close_lag1'] = X_data['Close'].shift(1)
X_data['Sentiment_lag1'] = X_data['Investor_Sentiment'].shift(1)

# 타겟 변수(y) 생성: 내일의 주가(수익률)
X_data['Target_Close'] = X_data['Close'].shift(-1) 

# 결측치 제거
X_data = X_data.dropna()

# 최종 X와 y 분리
X = X_data.drop(columns=['Target_Close'])
y = X_data['Target_Close']


# =========================================================
# 4. 데이터 분할 (Train 70% / Val 15% / Test 15%)
# =========================================================
print("\n=========================================================")
print(" 🧠 데이터 분할 및 XGBoost 모델 학습")
print("=========================================================")

# Step 1: 전체 데이터 중 앞 70%를 Train으로, 뒤 30%를 Temp로 나눔
X_train_main, X_temp, y_train_main, y_temp = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# Step 2: 남은 Temp(30%)를 정확히 반으로 나누어 Val(15%)과 Test(15%)로 분할
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)

print(f"📊 [데이터 분할 결과 (시간 순서 보존)]")
print(f" - Train Set: {len(X_train_main)}건 (학습용)")
print(f" - Val Set  : {len(X_val)}건 (검증 및 조기종료용)")
print(f" - Test Set : {len(X_test)}건 (최종 성능 평가용)\n")


# =========================================================
# 5. XGBoost 모델 정의 및 학습 (Early Stopping)
# =========================================================
model = XGBRegressor(
    n_estimators=1000,          # 최대 트리 개수
    learning_rate=0.01,         # 학습률
    max_depth=4,                # 트리의 깊이
    early_stopping_rounds=50,   # Val 오차가 50번 동안 안 줄면 학습 중단
    random_state=42
)

print("🚀 모델 학습 및 검증 시작 (Early Stopping 작동 중)...")
# eval_set에 Train과 Val을 순서대로 넣어서 성능을 모니터링합니다.
model.fit(
    X_train_main, y_train_main,
    eval_set=[(X_train_main, y_train_main), (X_val, y_val)],
    verbose=False
)

print(f"✅ 학습 완료! (조기 종료 시점: {model.best_iteration}번째 트리)")


# =========================================================
# 6. 아예 본 적 없는 Test 데이터로 최종 성능 평가
# =========================================================
y_pred = model.predict(X_test)

final_r2 = r2_score(y_test, y_pred)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_mae = mean_absolute_error(y_test, y_pred)

print("\n📊 [최종 Test 데이터 평가 결과]")
print(f'R² Score : {final_r2:.4f}')
print(f'RMSE     : {final_rmse:.4f}')
print(f'MAE      : {final_mae:.4f}')

# 모델 저장
joblib.dump(model, 'best_xgboost_model.pkl')
print("\n💾 모델 저장 완료: 'best_xgboost_model.pkl'")