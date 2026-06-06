# check_data_status.py
"""
Phase 3-1단계: 데이터 준비 상태 확인
당신의 폴더 구조 기반으로 파일 존재 여부와 데이터 스키마 확인
"""

import os
import sys
from pathlib import Path
import pandas as pd

print("\n" + "="*80)
print("🔍 DATA PREPARATION STATUS CHECK - Phase 3-1단계")
print("="*80 + "\n")

# 현재 작업 디렉토리
work_dir = Path.cwd()
print(f"📂 작업 디렉토리: {work_dir}\n")

# ======================== PART 1: 폴더 구조 확인 ========================
print("┌─ PART 1: 폴더 구조 확인")
print("├" + "─"*78)

required_dirs = {
    'data': 'data/',
    'data/cache': 'data/cache/',
    'data/output': 'data/output/',
    'models': 'models/',
    'src': 'src/',
}

dir_status = {}
for name, path in required_dirs.items():
    full_path = work_dir / path
    exists = full_path.exists()
    dir_status[name] = exists
    status_icon = "✅" if exists else "❌"
    print(f"│ {status_icon} {path}")

print("└" + "─"*78 + "\n")

# ======================== PART 2: 캐시 데이터 파일 확인 ========================
print("┌─ PART 2: 캐시 데이터 파일 확인 (data/cache/)")
print("├" + "─"*78)

cache_files = {
    'Samsung': 'data/cache/Samsung_sentiment_proxy.csv',
    'KOSPI': 'data/cache/KOSPI_sentiment_proxy.csv',
    'Bitcoin': 'data/cache/Bitcoin_sentiment_proxy.csv',
    'Meta': 'data/cache/meta.csv'
}

cache_status = {}
for asset_name, file_path in cache_files.items():
    full_path = work_dir / file_path
    exists = full_path.exists()
    cache_status[asset_name] = exists
    status_icon = "✅" if exists else "❌"
    
    if exists:
        try:
            df = pd.read_csv(full_path)
            shape_info = f"  {df.shape[0]} rows × {df.shape[1]} cols"
            print(f"│ {status_icon} {file_path}")
            print(f"│    ├─ 크기: {shape_info}")
            print(f"│    └─ 컬럼: {list(df.columns)[:5]}... (처음 5개)")
        except Exception as e:
            print(f"│ ⚠️  {file_path} (읽기 오류: {str(e)[:50]})")
    else:
        print(f"│ {status_icon} {file_path}")

print("└" + "─"*78 + "\n")

# ======================== PART 3: 모델 파일 확인 ========================
print("┌─ PART 3: 학습된 모델 파일 확인 (models/)")
print("├" + "─"*78)

model_files = {
    'XGBoost_ModelA': 'models/best_xgboost_panel_model_A.pkl',
    'XGBoost_ModelB': 'models/best_xgboost_panel_model_B.pkl',
    'XGBoost_ModelC': 'models/best_xgboost_panel_model_C.pkl',
    'PCA': 'models/pca.pkl',
    'Scaler': 'models/scaler.pkl'
}

model_status = {}
for model_name, file_path in model_files.items():
    full_path = work_dir / file_path
    exists = full_path.exists()
    model_status[model_name] = exists
    status_icon = "✅" if exists else "❌"
    
    if exists:
        file_size = os.path.getsize(full_path) / (1024*1024)  # MB
        print(f"│ {status_icon} {file_path}")
        print(f"│    └─ 크기: {file_size:.2f} MB")
    else:
        print(f"│ {status_icon} {file_path}")

print("└" + "─"*78 + "\n")

# ======================== PART 4: 출력 파일 확인 ========================
print("┌─ PART 4: 모델 결과 출력 파일 확인 (data/output/)")
print("├" + "─"*78)

output_files = {
    'Backtest_Summary': 'data/output/model_backtest_summary.csv',
    'Period_Backtest': 'data/output/model_period_backtest.csv',
    'Predictions': 'data/output/model_predictions.csv'
}

output_status = {}
for result_name, file_path in output_files.items():
    full_path = work_dir / file_path
    exists = full_path.exists()
    output_status[result_name] = exists
    status_icon = "✅" if exists else "❌"
    
    if exists:
        try:
            df = pd.read_csv(full_path)
            shape_info = f"  {df.shape[0]} rows × {df.shape[1]} cols"
            print(f"│ {status_icon} {file_path}")
            print(f"│    ├─ 크기: {shape_info}")
            print(f"│    └─ 컬럼: {list(df.columns)[:6]}... (처음 6개)")
        except Exception as e:
            print(f"│ ⚠️  {file_path} (읽기 오류: {str(e)[:50]})")
    else:
        print(f"│ {status_icon} {file_path}")

print("└" + "─"*78 + "\n")

# ======================== PART 5: 상세 데이터 스키마 ========================
print("┌─ PART 5: 데이터 스키마 상세 분석")
print("├" + "─"*78)

for asset_name, file_path in cache_files.items():
    if asset_name == 'Meta':
        continue  # Meta는 스킵
    
    full_path = work_dir / file_path
    if full_path.exists():
        print(f"│ [{asset_name}] {file_path}")
        try:
            df = pd.read_csv(full_path)
            print(f"│   Shape: {df.shape}")
            print(f"│   Date Range: {df['Date'].min()} ~ {df['Date'].max()}")
            print(f"│   Columns:")
            for col in df.columns:
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                print(f"│     - {col}: {dtype} (null: {null_count})")
            print(f"│   Sample (첫 3행):")
            print(f"│   {df.head(3).to_string()}")
        except Exception as e:
            print(f"│   ⚠️  오류: {str(e)}")
        print("│")

print("└" + "─"*78 + "\n")

# ======================== PART 6: 종합 점검 결과 ========================
print("┌─ PART 6: 종합 점검 결과")
print("├" + "─"*78)

all_ready = all([
    dir_status.get('data/cache', False),
    dir_status.get('data/output', False),
    dir_status.get('models', False),
    dir_status.get('src', False),
    all(cache_status.values()),
    all(model_status.values()),
])

if all_ready:
    print("│ ✅ 모든 준비가 완료되었습니다!")
    print("│ → 다음 단계: src/ 폴더에 visualization 스크립트 작성 가능")
else:
    print("│ ⚠️  일부 파일이 부족합니다:")
    print("│")
    
    if not cache_status.get('Samsung', False):
        print("│   ❌ Samsung_sentiment_proxy.csv 필요")
    if not cache_status.get('KOSPI', False):
        print("│   ❌ KOSPI_sentiment_proxy.csv 필요")
    if not cache_status.get('Bitcoin', False):
        print("│   ❌ Bitcoin_sentiment_proxy.csv 필요")
    if not model_status.get('XGBoost_ModelA', False):
        print("│   ❌ best_xgboost_panel_model_A.pkl 필요")
    if not model_status.get('XGBoost_ModelB', False):
        print("│   ❌ best_xgboost_panel_model_B.pkl 필요")
    if not model_status.get('XGBoost_ModelC', False):
        print("│   ❌ best_xgboost_panel_model_C.pkl 필요")
    if not model_status.get('PCA', False):
        print("│   ❌ pca.pkl 필요")
    if not model_status.get('Scaler', False):
        print("│   ❌ scaler.pkl 필요")
    if not output_status.get('Predictions', False):
        print("│   ❌ model_predictions.csv 필요")

print("└" + "─"*78 + "\n")

# ======================== 마지막: 요약 테이블 ========================
print("┌─ 요약 테이블")
print("├" + "─"*78)

summary_data = {
    '항목': ['디렉토리', '캐시 데이터', '모델 파일', '출력 결과'],
    '상태': [
        '✅' if all(dir_status.values()) else '❌',
        '✅' if all(cache_status.values()) else '❌',
        '✅' if all(model_status.values()) else '❌',
        '✅' if all(output_status.values()) else '❌',
    ]
}

for item, status in zip(summary_data['항목'], summary_data['상태']):
    print(f"│ {status} {item}")

print("└" + "─"*78 + "\n")

print("="*80)
print("✨ 점검 완료! 위 결과를 확인하고 다음 단계로 진행하세요.")
print("="*80 + "\n")
