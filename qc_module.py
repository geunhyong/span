import pandas as pd
import numpy as np
import glob
import os

def load_all_csv(directory_path):
    """지정된 디렉토리 내의 모든 CSV 파일을 읽어서 하나로 합침"""
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    if not all_files:
        return pd.DataFrame()
    
    df_list = []
    for filename in all_files:
        # 1. 인코딩(cp949)을 명시하고, '일시' 컬럼을 날짜 형식으로 파싱합니다.
        # (만약 한글이 깨지거나 에러가 나면 encoding='euc-kr' 로 변경해 보세요)
        df = pd.read_csv(filename, parse_dates=['일시'], encoding='cp949') 
        
        # 2. 코드 내부에서 사용할 수 있도록 컬럼명을 영문으로 변경합니다.
        # '일시' -> 'datetime', '기온(°C)' -> 'temp'
        df = df.rename(columns={'일시': 'datetime', '기온(°C)': 'temp'})
        
        df_list.append(df)
    
    # 3. 여러 개의 데이터프레임을 하나로 합치고, 시간 순으로 정렬 후 인덱스 설정
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    full_df = full_df.sort_values('datetime').set_index('datetime')
    
    return full_df

def apply_qc_logic(df, physical_range=(-40, 60), step_limit=3.0, consistency_limit=0.1):
    """5단계 QC 로직 적용"""
    qc_df = df.copy()
    
    # 1. 결측 검사 (기존에 NaN이 아니더라도 빈 값은 nan 처리)
    qc_df['temp'] = pd.to_numeric(qc_df['temp'], errors='coerce')

    # 2. 물리 한계 검사
    p_min, p_max = physical_range
    qc_df.loc[(qc_df['temp'] < p_min) | (qc_df['temp'] > p_max), 'temp'] = np.nan

    # 3. 단계 검사 (현재 - 1분전)
    # diff 계산 시 결측치가 있으면 그 다음 유효 데이터와의 차이를 계산하지 않도록 유의
    temp_diff = qc_df['temp'].diff() #.abs() 계산은 diff 이후에 수행하여 절대값으로 비교
    qc_df.loc[temp_diff.abs() > step_limit, 'temp'] = np.nan

    # 4. 지속성 검사 (60분간 변화량 절대값의 합)
    # 1시간 단위로 그룹화하여 검사
    diff_abs = qc_df['temp'].diff().abs()
    # 60개(1시간)씩 롤링 윈도우를 사용하여 합산
    rolling_sum = diff_abs.rolling(window=60, min_periods=60).sum()
    
    # 합계가 0.1보다 작으면 해당 구간(60개) 전체 오류 처리
    # (실제 기상 업무 알고리즘에 따라 현재 시점부터 역으로 60개를 지울지 결정)
    bad_indices = rolling_sum[rolling_sum < consistency_limit].index
    for idx in bad_indices:
        # 해당 종료 시점부터 59분 전까지를 모두 nan으로
        start_idx = idx - pd.Timedelta(minutes=59)
        qc_df.loc[start_idx:idx, 'temp'] = np.nan

    return qc_df

def calc_mean_with_80_rule(series, expected_count):
    """유효 데이터가 80% 미만이면 NaN 반환"""
    valid_count = series.count()
    if expected_count <= 0 or (valid_count / expected_count) <= 0.8:
        return np.nan
    return series.mean()

def get_aggregated_data(df):
    """시간대별 평균 산출 (1h, 3h, 8h, 24h)"""
    results = {}
    # 1분 단위 데이터 가정: 1시간=60개, 3시간=180개, 8시간=480개, 1일=1440개
    results['1H'] = df['temp'].resample('1h').apply(lambda x: calc_mean_with_80_rule(x, 60))
    results['3H'] = df['temp'].resample('3h').apply(lambda x: calc_mean_with_80_rule(x, 180))
    results['8H'] = df['temp'].resample('8h').apply(lambda x: calc_mean_with_80_rule(x, 480))
    results['1D'] = df['temp'].resample('1D').apply(lambda x: calc_mean_with_80_rule(x, 1440))
    return results

def analyze_commute_time(df):
    """
    평일(월~금) 출근시간(06:00~08:59)과 주말 같은 시간 데이터를 추출하여 분석
    """
    # 1. 요일 마스크 (0:월, 1:화, 2:수, 3:목, 4:금, 5:토, 6:일)
    is_weekday = df.index.dayofweek < 5
    is_weekend = df.index.dayofweek >= 5
    
    # 2. 시간 마스크 (6시 00분 ~ 8시 59분)
    is_morning_rush = (df.index.hour >= 6) & (df.index.hour < 9)
    
    # 3. 데이터 분리
    weekday_rush_df = df[is_weekday & is_morning_rush]
    weekend_rush_df = df[is_weekend & is_morning_rush]
    
    return weekday_rush_df, weekend_rush_df