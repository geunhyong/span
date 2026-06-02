
import pandas as pd

def load_data(filename):
    """
    데이터를 로드하는 함수
    :param filename: 파일 이름
    :return: 데이터프레임
    """
    return pd.read_csv(filename)
