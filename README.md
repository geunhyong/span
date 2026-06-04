# Investor Sentiment Stock Dashboard

주봉 가격 데이터에서 기술지표를 계산하고, 가격 요인으로 설명되지 않는 잔차를 투자자 심리의 대리변수로 사용해 다음 주 로그수익률 방향을 검증하는 Streamlit 대시보드입니다.

## 실행

```powershell
streamlit run dashboard.py
```

필요 패키지는 `requirements.txt`에 정리되어 있습니다.

## 모델 생성

대시보드는 모델을 새로 학습하지 않고 `models/` 폴더에 저장된 pkl 파일만 사용합니다.

```powershell
python model_train.py
```

생성되는 파일:

- `models/best_xgboost_panel_model_A.pkl`
- `models/best_xgboost_panel_model_B.pkl`
- `models/best_xgboost_panel_model_C.pkl`
- `models/pca.pkl`
- `models/scaler.pkl`

## 파일 구조

```text
.
├── dashboard.py
├── visualization.py
├── tabs
│   ├── data_preprocessing.py
│   ├── sentiment_proxy.py
│   └── modeling_validation.py
├── utils
│   ├── performance_metrics.py
│   ├── data_utils.py
│   └── plotting_utils.py
└── log
    ├── introduction.py
    └── process_flow.py
```

## 역할

- `dashboard.py`: Streamlit 메인 UI와 탭 라우팅
- `tabs/data_preprocessing.py`: 데이터 수집, 지표 계산 결과 확인
- `tabs/sentiment_proxy.py`: OLS 잔차 추출, PCA 심리지수 생성
- `tabs/modeling_validation.py`: XGBoost 학습 및 성능 검증
- `visualization.py`: 공통 시각화 함수
- `utils/performance_metrics.py`: RMSE, MAE, R², 방향 정확도 계산
- `utils/data_utils.py`: CSV I/O와 컬럼 검증
- `utils/plotting_utils.py`: 공통 차트 스타일

## 사용 흐름

1. 사이드바에서 분석 대상을 선택합니다.
2. `데이터 전처리` 탭에서 최근 주봉 데이터를 불러옵니다.
3. `심리지표` 탭에서 OLS 잔차와 PCA 심리지수를 계산합니다.
4. `모델링/검증` 탭에서 XGBoost 모델을 학습하고 성능을 확인합니다.
