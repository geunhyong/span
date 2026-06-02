
import streamlit as st
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (
    r2_score,
    root_mean_squared_error,
    mean_absolute_error)

def run():
    st.header("**모델링 및 검증**")
    
    # 데이터 적재 및 훈련, 테스트 분할
    data = pd.read_csv('processed_data.csv')  # 미리 전처리된 데이터 로드
    st.subheader("데이터 확인") 
    st.write(data.head())
    st.write("*-=*-=*-=*-=*-=*"*3)
    st.subheader("모델링에 사용할 피처와 타겟 변수 설정(Feature and Target Selection)")   
    X = data[['ATR', 'MFI', 'Stochastic']]  # 피처
    y = data['Close']  # 타겟
    
    st.subheader("시계열 Train_Test_Split(마지막 fold 사용)"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False) #random_state has no effect when shuffle=False
    
    # XGBoost 모델 훈련
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    
    # 예측 및 성능 평가
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    st.subheader("모델 성능")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.2f}")
