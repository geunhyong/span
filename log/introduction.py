# log/introduction.py
def write_introduction():
    introduction = """
    본 프로젝트에서는 주가 예측을 위한 심리지수를 모델링하고 평가합니다.
    데이터 수집부터 진행 데이터 시각화까지, 프로젝트의 모든 단계를 개괄적으로 설명합니다.
    """
    with open('introduction.md', 'w') as f:
        f.write(introduction)
