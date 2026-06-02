# Stock Price Prediction Sentiment Index Modeling and Evaluation
## 프로젝트 개요
이 프로젝트는 삼성전자(Samsung Electronics), 비트코인(Bitcoin), KOSPI 및 KOSDAQ의 시장 연결성을 조사하고, 6년간의 주간 데이터를 분석하여 투자자 심리를 수치화하는 것을 목표로 합니다. 이를 통해 XGBoost 기반의 가격 예측을 개선하고자 합니다.
## 폴더 구조
StockPricePrediction/
├── dashboard.py                   # Streamlit을 이용한 대시보드
├── visualization.py               # 데이터 시각화 함수 모음
├── tabs/                          # 기능별 모듈
│   ├── data_preprocessing.py      # 데이터 전처리
│   ├── sentiment_proxy.py         # 심리지표 계산
│   ├── modeling_validation.py      # 모델 학습 및 검증
├── utils/                         # 공통 기능 모듈
│   ├── performance_metrics.py      # 성능 지표 계산
│   ├── data_utils.py              # CSV 입출력
│   ├── plotting_utils.py           # 시각화 관련 공통 함수
└── log/                           # 문서화 관련 파일
    ├── introduction.py            # 프로젝트 소개
    └── process_flow.py            # 작업 흐름 설명
## 설치 방법
1. 리포지토리를 클론합니다:
   ```bash
   git clone [repository-url]
2. 필요한 패키지를 설치 합니다
pip install -r requirements.txt
3. 대시보드를 실행 합니다
streamlit run dashboard.py
## 저자
- 주원 추 (리더)
- 민호 김
- 보성 권
- 근형 김

