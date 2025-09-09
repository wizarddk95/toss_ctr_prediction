"""
테스트 세트 Word2Vec 임베딩 앙상블 생성
- 5개 폴드 모델로 각각 임베딩 생성
- 폴드별 임베딩 저장
- 최종 앙상블(평균) 생성
"""

import os
import gc
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import List, Dict, Optional
from gensim.models import Word2Vec


# =========================
# 설정
# =========================
# 파일 경로
TEST_COMPRESSED_PATH = './data/seq_compression/test_seq_compressed.parquet'
MODEL_DIR = './models/w2v'
OUTPUT_DIR = './data/seq_w2v_embedding/test'

# 학습 시와 동일한 설정
SEQ_COL = 'seq_compressed'
BATCH_SIZE = 200_000
VECTOR_SIZE = 64
LAST_K_LIST = (5, 20, 50)
N_SPLITS = 5

# 정밀도 설정 (학습 시와 동일)
FLOAT_PRECISION = 4
USE_FLOAT16 = False

# 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# 유틸 함수
# =========================
def reduce_precision(arr: np.ndarray, precision: int = 4, use_float16: bool = False) -> np.ndarray:
    """정밀도 감소"""
    arr_rounded = np.round(arr, precision)
    return arr_rounded.astype(np.float16) if use_float16 else arr_rounded.astype(np.float32)


def process_tokens_to_embeddings(tokens: List[str], 
                                w2v: Word2Vec, 
                                last_k_list: tuple,
                                precision: int = 4,
                                use_float16: bool = False) -> Dict:
    """토큰 리스트를 임베딩으로 변환"""
    vecs = []
    oov_count = 0
    
    # 토큰을 벡터로 변환
    for token in tokens:
        if token in w2v.wv:
            vecs.append(w2v.wv[token])
        else:
            oov_count += 1
    
    d = w2v.vector_size
    
    if not vecs:
        # 모든 토큰이 OOV인 경우
        zero = np.zeros(d, dtype=np.float32)
        reps = {'mean': zero, 'last': zero, 'max': zero}
        for k in last_k_list:
            reps[f'last{k}'] = zero
    else:
        V = np.vstack(vecs).astype(np.float32)
        
        # 다양한 대표 임베딩 계산
        reps = {
            'mean': V.mean(axis=0),
            'last': V[-1],
            'max': V.max(axis=0)
        }
        
        # Last-K 임베딩
        for k in last_k_list:
            if len(V) >= k:
                reps[f'last{k}'] = V[-k:].mean(axis=0)
            else:
                reps[f'last{k}'] = V.mean(axis=0)
    
    # 정밀도 감소
    for name in reps:
        reps[name] = reduce_precision(reps[name], precision, use_float16)
    
    # 딕셔너리로 펼치기
    rec = {}
    for name, vec in reps.items():
        for i in range(vec.shape[0]):
            rec[f'{name}_{i}'] = float(vec[i])
    
    # OOV 비율 추가
    rec['oov_ratio'] = oov_count / len(tokens) if tokens else 0.0
    
    return rec


def generate_test_embeddings_single_fold(
    test_path: str,
    model_path: str,
    output_path: str,
    fold: int,
    batch_size: int = 200_000
) -> bool:
    """
    단일 폴드 모델로 테스트 임베딩 생성
    
    Returns:
        성공 여부
    """
    print(f"\n--- Fold {fold} ---")
    
    # 모델 로드
    if not os.path.exists(model_path):
        print(f"  ❌ Model not found: {model_path}")
        return False
    
    print(f"  Loading model: {model_path}")
    w2v_model = Word2Vec.load(model_path)
    vocab_size = len(w2v_model.wv)
    print(f"  Vocabulary size: {vocab_size:,}")
    
    # 테스트 파일 처리
    print(f"  Processing test file...")
    writer = None
    row_id = 0
    total_oov = 0
    total_tokens = 0
    
    try:
        pf = pq.ParquetFile(test_path)
        total_rows = pf.metadata.num_rows
        print(f"  Total test rows: {total_rows:,}")
        
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=[SEQ_COL])):
            df_batch = batch.to_pandas()
            
            # 토큰화
            tokens_series = df_batch[SEQ_COL].fillna('').str.split(',')
            
            # 각 토큰 리스트 정제
            tokens_series = tokens_series.apply(
                lambda toks: [t.strip() for t in toks if t.strip()]
            )
            
            # 임베딩 생성
            embedding_records = tokens_series.apply(
                lambda tokens: process_tokens_to_embeddings(
                    tokens, w2v_model, LAST_K_LIST, FLOAT_PRECISION, USE_FLOAT16
                )
            )
            
            # DataFrame 생성
            df = pd.DataFrame.from_records(embedding_records.tolist())
            df['row_id'] = range(row_id, row_id + len(df))
            row_id += len(df)
            
            # OOV 통계 수집
            total_oov += df['oov_ratio'].sum() * len(df)
            total_tokens += len(df)
            
            # float16 변환
            if USE_FLOAT16:
                float_cols = [col for col in df.columns if col not in ['row_id', 'oov_ratio']]
                for col in float_cols:
                    df[col] = df[col].astype(np.float16)
            
            # 파일 저장
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)
            
            # 진행 상황
            if batch_idx % 5 == 0:
                progress = (row_id / total_rows) * 100
                print(f"    Progress: {progress:.1f}% ({row_id:,}/{total_rows:,})", end='\r')
        
        print()  # 줄바꿈
        
    finally:
        if writer:
            writer.close()
    
    # 통계 출력
    avg_oov = (total_oov / total_tokens) * 100 if total_tokens > 0 else 0
    print(f"  ✅ Saved: {output_path}")
    print(f"  Average OOV rate: {avg_oov:.2f}%")
    
    # 메모리 정리
    del w2v_model
    gc.collect()
    
    return True


def create_ensemble_average(fold_paths: List[str], output_path: str):
    """
    여러 폴드의 임베딩을 평균내어 앙상블 생성
    """
    print("\n" + "="*70)
    print("Creating Ensemble (Average)")
    print("="*70)
    
    # 유효한 폴드 파일만 필터링
    valid_paths = [p for p in fold_paths if os.path.exists(p)]
    
    if len(valid_paths) == 0:
        print("❌ No fold embeddings found!")
        return
    
    if len(valid_paths) < len(fold_paths):
        print(f"⚠️ Warning: Only {len(valid_paths)}/{len(fold_paths)} fold files found")
    
    print(f"Averaging {len(valid_paths)} fold embeddings...")
    
    # 첫 번째 파일로 구조 확인
    first_df = pd.read_parquet(valid_paths[0])
    n_rows = len(first_df)
    print(f"  Rows per fold: {n_rows:,}")
    
    # 배치 단위로 처리 (메모리 효율)
    batch_size = 50000
    n_batches = (n_rows + batch_size - 1) // batch_size
    
    writer = None
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_rows)
        
        print(f"  Processing batch {batch_idx+1}/{n_batches} (rows {start_idx:,}-{end_idx:,})...", end='\r')
        
        # 각 폴드에서 배치 읽기
        batch_dfs = []
        for path in valid_paths:
            df = pd.read_parquet(path).iloc[start_idx:end_idx]
            df = df.sort_values('row_id').reset_index(drop=True)
            batch_dfs.append(df)
        
        # row_id와 기타 메타 정보는 첫 번째 폴드에서 가져옴
        ensemble_df = batch_dfs[0][['row_id']].copy()
        
        # OOV ratio는 평균
        if 'oov_ratio' in batch_dfs[0].columns:
            oov_ratios = np.stack([df['oov_ratio'].values for df in batch_dfs])
            ensemble_df['oov_ratio'] = oov_ratios.mean(axis=0)
        
        # 임베딩 컬럼들 평균
        embedding_cols = [col for col in batch_dfs[0].columns 
                         if col not in ['row_id', 'oov_ratio']]
        
        for col in embedding_cols:
            col_values = np.stack([df[col].values for df in batch_dfs])
            mean_values = col_values.mean(axis=0)
            
            # 정밀도 감소
            if FLOAT_PRECISION:
                mean_values = np.round(mean_values, FLOAT_PRECISION)
            if USE_FLOAT16:
                mean_values = mean_values.astype(np.float16)
            
            ensemble_df[col] = mean_values
        
        # 파일 저장
        table = pa.Table.from_pandas(ensemble_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
        writer.write_table(table)
        
        # 메모리 정리
        del batch_dfs, ensemble_df
        gc.collect()
    
    print()  # 줄바꿈
    
    if writer:
        writer.close()
    
    print(f"✅ Ensemble saved: {output_path}")
    
    # 파일 크기 정보
    for path in valid_paths + [output_path]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  - {os.path.basename(path)}: {size_mb:.1f} MB")


def verify_embeddings(fold_paths: List[str], ensemble_path: str, sample_size: int = 5):
    """
    생성된 임베딩 검증
    """
    print("\n" + "="*70)
    print("Verification")
    print("="*70)
    
    # 앙상블 파일 확인
    if os.path.exists(ensemble_path):
        ensemble_df = pd.read_parquet(ensemble_path)
        print(f"\nEnsemble shape: {ensemble_df.shape}")
        print(f"Columns: {ensemble_df.shape[1]} features")
        print(f"Memory usage: {ensemble_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # 샘플 통계
        print(f"\nSample statistics (first {sample_size} rows):")
        sample = ensemble_df.head(sample_size)
        
        # 임베딩 컬럼들의 통계
        embedding_cols = [col for col in ensemble_df.columns 
                         if col not in ['row_id', 'oov_ratio']]
        
        if embedding_cols:
            print(f"  Mean embedding values:")
            for i in range(min(3, len(embedding_cols))):
                col = embedding_cols[i]
                print(f"    {col}: {ensemble_df[col].mean():.4f} (±{ensemble_df[col].std():.4f})")
        
        if 'oov_ratio' in ensemble_df.columns:
            print(f"\n  OOV statistics:")
            print(f"    Mean: {ensemble_df['oov_ratio'].mean():.3f}")
            print(f"    Max:  {ensemble_df['oov_ratio'].max():.3f}")
            print(f"    Min:  {ensemble_df['oov_ratio'].min():.3f}")
    
    # 폴드 간 일관성 확인
    print(f"\nFold consistency check:")
    valid_paths = [p for p in fold_paths if os.path.exists(p)]
    
    if len(valid_paths) >= 2:
        # 첫 번째 행의 몇 개 값 비교
        row_0_values = []
        for path in valid_paths[:2]:
            df = pd.read_parquet(path, columns=['mean_0', 'mean_1', 'mean_2']).head(1)
            values = df[['mean_0', 'mean_1', 'mean_2']].values[0]
            row_0_values.append(values)
            print(f"  {os.path.basename(path)}: {values}")
        
        # 차이 계산
        if len(row_0_values) == 2:
            diff = np.abs(row_0_values[0] - row_0_values[1])
            print(f"  Absolute difference: {diff}")


# =========================
# 메인 실행
# =========================
def main():
    print("="*70)
    print("TEST SET WORD2VEC EMBEDDING GENERATION")
    print("="*70)
    
    # 설정 출력
    print(f"\nConfiguration:")
    print(f"  Test file: {TEST_COMPRESSED_PATH}")
    print(f"  Model directory: {MODEL_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Number of folds: {N_SPLITS}")
    print(f"  Precision: {FLOAT_PRECISION} decimals")
    print(f"  Data type: {'float16' if USE_FLOAT16 else 'float32'}")
    
    # 파일 확인
    if not os.path.exists(TEST_COMPRESSED_PATH):
        print(f"\n❌ Error: Test file not found: {TEST_COMPRESSED_PATH}")
        return
    
    # 테스트 파일 정보
    pf = pq.ParquetFile(TEST_COMPRESSED_PATH)
    test_rows = pf.metadata.num_rows
    print(f"\nTest data: {test_rows:,} rows")
    
    print("\n" + "="*70)
    print("STEP 1: Generate embeddings for each fold")
    print("="*70)
    
    # 각 폴드별로 임베딩 생성
    fold_paths = []
    successful_folds = 0
    
    for fold in range(N_SPLITS):
        model_path = os.path.join(MODEL_DIR, f'w2v_fold{fold}.model')
        output_path = os.path.join(OUTPUT_DIR, f'test_fold{fold}.parquet')
        
        success = generate_test_embeddings_single_fold(
            test_path=TEST_COMPRESSED_PATH,
            model_path=model_path,
            output_path=output_path,
            fold=fold,
            batch_size=BATCH_SIZE
        )
        
        if success:
            fold_paths.append(output_path)
            successful_folds += 1
        else:
            print(f"  ⚠️ Skipping fold {fold}")
    
    print(f"\n✅ Generated embeddings for {successful_folds}/{N_SPLITS} folds")
    
    if successful_folds == 0:
        print("❌ No embeddings generated. Exiting.")
        return
    
    # 앙상블 생성
    print("\n" + "="*70)
    print("STEP 2: Create ensemble (average)")
    print("="*70)
    
    ensemble_path = os.path.join(OUTPUT_DIR, 'test_ensemble.parquet')
    create_ensemble_average(fold_paths, ensemble_path)
    
    # 검증
    verify_embeddings(fold_paths, ensemble_path)
    
    # 완료 메시지
    print("\n" + "="*70)
    print("✅ COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\nGenerated files:")
    print(f"  Individual folds:")
    for fold in range(N_SPLITS):
        path = os.path.join(OUTPUT_DIR, f'test_fold{fold}.parquet')
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"    - test_fold{fold}.parquet ({size_mb:.1f} MB)")
    
    print(f"\n  Ensemble:")
    if os.path.exists(ensemble_path):
        size_mb = os.path.getsize(ensemble_path) / (1024 * 1024)
        print(f"    - test_ensemble.parquet ({size_mb:.1f} MB)")
    
    # 사용 방법
    print(f"\n📝 Usage example:")
    print(f"```python")
    print(f"import pandas as pd")
    print(f"")
    print(f"# Load ensemble embeddings")
    print(f"test_embeddings = pd.read_parquet('{ensemble_path}')")
    print(f"")
    print(f"# Or load specific fold")
    print(f"fold0_embeddings = pd.read_parquet('{OUTPUT_DIR}/test_fold0.parquet')")
    print(f"```")


if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"\n⏱️ Total execution time: {elapsed:.1f} seconds")