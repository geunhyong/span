# log/process_flow.py
def log_process_flow():
    process_flow = """
    **프로세스 흐름**
    1. 데이터 수집
    2. 데이터 전처리
    3. 심리지표 계산
    4. 모델링 및 검증
    5. 성능 분석
    6. 결과 시각화
    """
    with open('process_flow.md', 'w') as f:
        f.write(process_flow)
