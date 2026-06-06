"""
sentiment_proxy_rebuilt.csv의 중복 컬럼 제거 및 정리
"""
import pandas as pd
import os

def clean_sentiment_csv():
    """CSV 파일의 중복 컬럼 제거"""
    
    csv_path = 'data/output/sentiment_proxy_rebuilt.csv'
    df = pd.read_csv(csv_path, index_col=0)
    
    print(f"🔍 원본 shape: {df.shape}")
    print(f"원본 컬럼 수: {len(df.columns)}\n")
    
    # 중복 컬럼 제거 (.1 suffix가 있는 것들)
    cols_to_drop = [col for col in df.columns if col.endswith('.1')]
    
    if cols_to_drop:
        print(f"🗑️ 제거할 컬럼 ({len(cols_to_drop)}개): {cols_to_drop}")
        df_clean = df.drop(columns=cols_to_drop)
    else:
        print("✅ 중복 컬럼 없음")
        df_clean = df
    
    print(f"✅ 정리 후 shape: {df_clean.shape}\n")
    
    # 저장
    df_clean.to_csv(csv_path)
    print(f"✅ 저장 완료: {csv_path}")
    
    # 검증
    print(f"\n[검증] 최종 컬럼 리스트:")
    for i, col in enumerate(df_clean.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return df_clean

if __name__ == "__main__":
    clean_sentiment_csv()
