"""
안전한 Sentiment Proxy 재구성
- 기존 데이터 활용
- 새로운 폴더에 저장
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_samsung_data():
    """삼성전자 데이터 로드"""
    try:
        df = pd.read_csv(
            'data/cache/삼성전자_sentiment_proxy.csv',
            index_col=0
        )
        df.index = pd.to_datetime(df.index, errors='coerce')
        return df.sort_index()
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return None

def extract_residuals_safe(df, target_col, predictor_cols):
    """
    안전한 Residual 추출
    - NaN 처리
    - 에러 핸들링
    """
    try:
        # NaN 제거
        available_cols = [col for col in predictor_cols if col in df.columns]
        working_df = df[[target_col] + available_cols].dropna()
        
        if len(working_df) < 10:
            print(f"⚠️  {target_col}: 데이터 부족 ({len(working_df)}개)")
            return None
        
        X = working_df[available_cols]
        y = working_df[target_col]
        
        # OLS 회귀
        model = LinearRegression()
        model.fit(X, y)
        residuals = y - model.predict(X)
        
        return residuals
    except Exception as e:
        print(f"❌ {target_col} 추출 실패: {e}")
        return None

def rebuild_sentiment_proxy():
    """Step by Step Sentiment Proxy 재구성"""
    
    print("\n" + "="*70)
    print("🔧 SENTIMENT PROXY 재구성 (안전 모드)")
    print("="*70)
    
    # 1단계: 데이터 로드
    print("\n[1단계] 데이터 로드...")
    df = load_samsung_data()
    if df is None:
        return False
    print(f"✅ 로드됨: {df.shape}")
    print(f"   컬럼: {list(df.columns)[:5]}...")
    
    # 2단계: Residuals 추출
    print("\n[2단계] Technical Indicator Residuals 추출...")
    
    predictor_cols = [col for col in df.columns if 'Log_Return_lag' in col or col == 'Close']
    print(f"   예측변수: {predictor_cols}")
    
    residuals_dict = {}
    indicators = ['ATR_10', 'MFI_10', 'STOCHk_10_3_3']
    
    for indicator in indicators:
        if indicator in df.columns:
            res = extract_residuals_safe(df, indicator, predictor_cols)
            if res is not None:
                residuals_dict[f'{indicator}_res'] = res
                print(f"   ✅ {indicator}_res: {len(res)}개 추출")
            else:
                print(f"   ⚠️  {indicator}_res: 추출 실패")
    
    if not residuals_dict:
        print("❌ Residuals 추출 실패!")
        return False
    
    # 3단계: Residuals DataFrame 생성
    print("\n[3단계] Residuals DataFrame 생성...")
    residuals_df = pd.DataFrame(residuals_dict)
    print(f"✅ Shape: {residuals_df.shape}")
    print(f"   컬럼: {list(residuals_df.columns)}")
    
    # 4단계: Scaling
    print("\n[4단계] StandardScaler 적용...")
    scaler = StandardScaler()
    residuals_scaled = scaler.fit_transform(residuals_df)
    residuals_scaled_df = pd.DataFrame(
        residuals_scaled,
        columns=[f'{col}_scaled' for col in residuals_df.columns],
        index=residuals_df.index
    )
    print(f"✅ Scaled Shape: {residuals_scaled_df.shape}")
    print(f"   Mean: {residuals_scaled.mean(axis=0)}")
    print(f"   Std: {residuals_scaled.std(axis=0)}")
    
    # 5단계: PCA
    print("\n[5단계] PCA 적용...")
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(residuals_scaled)
    pc1_df = pd.DataFrame(
        pc1,
        columns=['Investor_Sentiment_PC1'],
        index=residuals_df.index
    )
    print(f"✅ PC1 생성 완료")
    print(f"   설명 분산: {pca.explained_variance_ratio_[0]:.4f}")
    print(f"   로딩: {pca.components_[0]}")
    
    # 6단계: 통합
    print("\n[6단계] 최종 통합...")
    result = pd.concat([df, residuals_df, residuals_scaled_df, pc1_df], axis=1)
    print(f"✅ 최종 Shape: {result.shape}")
    print(f"   새 컬럼: {list(result.columns)[-4:]}")
    
    # 7단계: 저장
    print("\n[7단계] 파일 저장...")
    
    # 데이터 저장
    output_path = 'data/output/sentiment_proxy_rebuilt.csv'
    result.to_csv(output_path)
    print(f"✅ {output_path} 저장됨")
    
    # Pickle 저장 (새 폴더)
    Path('models_new').mkdir(exist_ok=True)
    
    with open('models_new/scaler_new.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ models_new/scaler_new.pkl 저장됨")
    
    with open('models_new/pca_new.pkl', 'wb') as f:
        pickle.dump(pca, f)
    print(f"✅ models_new/pca_new.pkl 저장됨")
    
    # 메타데이터 저장
    metadata = {
        'shape': result.shape,
        'columns': list(result.columns),
        'residuals': list(residuals_dict.keys()),
        'pca_explained_variance': pca.explained_variance_ratio_[0],
        'pca_components': pca.components_[0].tolist(),
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }
    
    import json
    with open('models_new/sentiment_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ models_new/sentiment_metadata.json 저장됨")
    
    print("\n" + "="*70)
    print("✅ SENTIMENT PROXY 재구성 완료!")
    print("="*70)
    
    return result, scaler, pca

if __name__ == '__main__':
    result, scaler, pca = rebuild_sentiment_proxy()
    
    # 빠른 검증
    print("\n[검증]")
    print(f"Data Shape: {result.shape}")
    print(f"Sample rows:\n{result[['Close', 'ATR_10_res', 'Investor_Sentiment_PC1']].head()}")
