"""
개선된 Word2Vec OOF 임베딩 파이프라인
- 압축된 시퀀스 사용
- 라벨 파일 분리 처리
- 에러 핸들링 강화
"""

import os
import gc
import time
import multiprocessing
from typing import Iterable, List, Optional, Set, Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.model_selection import StratifiedKFold, KFold
from gensim.models import Word2Vec


# =========================
# 기본 하이퍼파라미터
# =========================
# 파일 경로 설정 (압축된 시퀀스와 원본 라벨 분리)
COMPRESSED_SEQ_PATH = './data/seq_compression/train_seq_compressed.parquet'
ORIGINAL_TRAIN_PATH = './data/train_optimized.parquet'  # 라벨용
SEQ_COL = 'seq_compressed'  # 압축된 시퀀스 컬럼명
LABEL_COL = 'clicked'

# 학습 설정
USE_STRATIFY = True
BATCH_ROWS = 200_000
VECTOR_SIZE = 64
WINDOW = 5
MIN_COUNT = 1
WORKERS = multiprocessing.cpu_count()
SG = 1  # 1: skip-gram, 0: CBOW
EPOCHS = 3
LAST_K_LIST = (5, 20, 50)
N_SPLITS = 5
SEED = 42

# 저장 정밀도 설정
FLOAT_PRECISION = 4
USE_FLOAT16 = False

# 출력 디렉토리
OUT_DIR = './data/seq_w2v_embedding'
MODEL_DIR = './models/w2v'
FOLD_ASSIGN_PATH = os.path.join(OUT_DIR, 'fold_assign.parquet')

# 디렉토리 생성
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# =========================
# 유틸: 말뭉치 이터레이터
# =========================
class CorpusIterator:
    """
    압축된 시퀀스를 배치 단위로 읽어 스트리밍
    """
    def __init__(self,
                 file_path: str,
                 batch_size: int = 100_000,
                 column: str = 'seq_compressed',
                 include_ids: Optional[Set[int]] = None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.column = column
        self.include_ids = include_ids

    def __iter__(self):
        global_row = 0
        pf = pq.ParquetFile(self.file_path)
        
        for batch in pf.iter_batches(batch_size=self.batch_size, columns=[self.column]):
            series = batch.to_pandas()[self.column]
            
            for idx, s in enumerate(series):
                take = (self.include_ids is None) or (global_row in self.include_ids)
                
                if take:
                    if isinstance(s, str) and s:
                        # 압축된 시퀀스 토큰화
                        tokens = [t.strip() for t in s.split(',') if t.strip()]
                        if tokens:  # 빈 시퀀스 제외
                            yield tokens
                    else:
                        # 빈 시퀀스는 건너뜀
                        pass
                
                global_row += 1


# =========================
# 라벨 및 폴드 관리
# =========================
def count_total_rows(file_path: str, batch_size: int = 200_000) -> int:
    """전체 행 수 계산"""
    pf = pq.ParquetFile(file_path)
    return pf.metadata.num_rows


def collect_labels(file_path: str,
                   label_col: str,
                   batch_size: int = 200_000) -> pd.DataFrame:
    """라벨 수집 (원본 파일에서)"""
    rows = []
    global_row = 0
    
    pf = pq.ParquetFile(file_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[label_col]):
        ser = batch.to_pandas()[label_col]
        cnt = len(ser)
        rows.append(pd.DataFrame({
            'row_id': np.arange(global_row, global_row + cnt, dtype=np.int64),
            'label': ser.values
        }))
        global_row += cnt
    
    return pd.concat(rows, ignore_index=True)


def load_or_make_folds() -> pd.DataFrame:
    """폴드 할당 로드 또는 생성"""
    if os.path.exists(FOLD_ASSIGN_PATH):
        print(f"[fold] Reusing existing fold assignment: {FOLD_ASSIGN_PATH}")
        assign = pd.read_parquet(FOLD_ASSIGN_PATH)
        assert {'row_id', 'fold'} <= set(assign.columns)
        return assign[['row_id', 'fold']].copy()

    print("[fold] Creating new fold assignment...")
    
    # 압축 파일과 원본 파일의 행 수 확인
    compressed_rows = count_total_rows(COMPRESSED_SEQ_PATH)
    
    if USE_STRATIFY and os.path.exists(ORIGINAL_TRAIN_PATH):
        # 원본 파일에서 라벨 가져오기
        original_rows = count_total_rows(ORIGINAL_TRAIN_PATH)
        
        if compressed_rows != original_rows:
            print(f"Warning: Row count mismatch! Compressed: {compressed_rows}, Original: {original_rows}")
        
        meta = collect_labels(ORIGINAL_TRAIN_PATH, LABEL_COL, BATCH_ROWS)
        y = meta['label'].values
        
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(meta['row_id'].values, y))
        
        assign = pd.DataFrame({'row_id': meta['row_id'].values, 'fold': -1}, dtype=np.int64)
        for f, (_, va_idx) in enumerate(splits):
            assign.loc[va_idx, 'fold'] = f
    else:
        # Stratify 없이 처리
        print("[fold] Using KFold without stratification")
        meta = pd.DataFrame({'row_id': np.arange(compressed_rows, dtype=np.int64)})
        
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(kf.split(meta['row_id'].values))
        
        assign = pd.DataFrame({'row_id': meta['row_id'].values, 'fold': -1}, dtype=np.int64)
        for f, (_, va_idx) in enumerate(splits):
            assign.loc[va_idx, 'fold'] = f

    assign.to_parquet(FOLD_ASSIGN_PATH, index=False)
    print(f"[fold] Saved: {FOLD_ASSIGN_PATH}")
    return assign


# =========================
# 임베딩 계산 함수들
# =========================
def reduce_precision(arr: np.ndarray, precision: int = 4, use_float16: bool = False) -> np.ndarray:
    """정밀도 감소"""
    arr_rounded = np.round(arr, precision)
    return arr_rounded.astype(np.float16) if use_float16 else arr_rounded.astype(np.float32)


def process_row_to_embeddings(tokens: List[str], 
                             w2v: Word2Vec, 
                             last_k_list: tuple,
                             precision: int = 4,
                             use_float16: bool = False) -> Dict:
    """토큰 리스트를 임베딩으로 변환"""
    vecs = []
    oov_count = 0  # Out-of-vocabulary 카운트
    
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
    
    # OOV 비율 추가 (선택적)
    rec['oov_ratio'] = oov_count / len(tokens) if tokens else 0.0
    
    return rec


def cache_fold_embeddings(file_path: str,
                         w2v: Word2Vec,
                         include_ids: Set[int],
                         out_path: str,
                         batch_size: int = 200_000,
                         seq_col: str = 'seq_compressed',
                         last_k_list: tuple = (5, 20, 50),
                         precision: int = 4,
                         use_float16: bool = False):
    """폴드별 임베딩 캐싱"""
    print(f"  Generating embeddings for {len(include_ids):,} rows...")
    
    writer = None
    all_row_ids = np.array(sorted(list(include_ids)))
    row_cursor = 0
    processed = 0
    
    pf = pq.ParquetFile(file_path)
    
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=[seq_col])):
        batch_start = row_cursor
        batch_end = row_cursor + len(batch)
        
        # 현재 배치에서 처리할 row_id 찾기
        target_mask = (all_row_ids >= batch_start) & (all_row_ids < batch_end)
        target_ids = all_row_ids[target_mask]
        
        if len(target_ids) > 0:
            ser = batch.to_pandas()[seq_col]
            
            # 대상 행만 필터링
            local_indices = target_ids - batch_start
            target_ser = ser.iloc[local_indices]
            
            # 토큰화
            tokens_series = target_ser.fillna('').str.split(',')
            
            # 임베딩 계산
            embedding_records = tokens_series.apply(
                lambda toks: process_row_to_embeddings(
                    [t.strip() for t in toks if t.strip()],
                    w2v, last_k_list, precision, use_float16
                )
            )
            
            # DataFrame 생성
            df = pd.DataFrame.from_records(embedding_records.tolist())
            df['row_id'] = target_ids
            
            # float16 변환 (선택적)
            if use_float16:
                float_cols = [col for col in df.columns if col not in ['row_id', 'oov_ratio']]
                for col in float_cols:
                    df[col] = df[col].astype(np.float16)
            
            # 파일 저장
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
            writer.write_table(table)
            
            processed += len(target_ids)
            
            # 진행 상황
            if batch_idx % 10 == 0:
                print(f"    Processed: {processed:,}/{len(include_ids):,} rows", end='\r')
        
        row_cursor = batch_end
    
    if writer:
        writer.close()
    
    print(f"    Completed: {processed:,} rows → {out_path}")


# =========================
# 메인 파이프라인
# =========================
def main():
    print("="*70)
    print("WORD2VEC OOF EMBEDDING PIPELINE")
    print("="*70)
    
    # 설정 출력
    print(f"\nConfiguration:")
    print(f"  - Compressed sequence file: {COMPRESSED_SEQ_PATH}")
    print(f"  - Vector size: {VECTOR_SIZE}")
    print(f"  - Window: {WINDOW}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Skip-gram: {SG == 1}")
    print(f"  - N splits: {N_SPLITS}")
    print(f"  - Float16: {USE_FLOAT16}")
    print(f"  - Precision: {FLOAT_PRECISION} decimals")
    
    # 파일 확인
    if not os.path.exists(COMPRESSED_SEQ_PATH):
        print(f"\n❌ Error: Compressed sequence file not found: {COMPRESSED_SEQ_PATH}")
        return
    
    # 폴드 할당
    assign = load_or_make_folds()
    total_rows = assign.shape[0]
    print(f"\n[Data] Total rows: {total_rows:,}")
    
    # 저장 공간 예측
    if USE_FLOAT16:
        print(f"[Storage] Using float16 will reduce storage by ~50%")
    
    print("\n" + "="*70)
    
    # 각 폴드 처리
    for fold in range(N_SPLITS):
        start_time = time.time()
        
        va_ids = set(assign.loc[assign['fold'] == fold, 'row_id'].tolist())
        tr_ids = set(assign.loc[assign['fold'] != fold, 'row_id'].tolist())
        
        print(f"\n===== FOLD {fold}/{N_SPLITS-1} =====")
        print(f"  Train: {len(tr_ids):,} rows")
        print(f"  Valid: {len(va_ids):,} rows")
        
        # Word2Vec 학습
        print(f"\n[Training Word2Vec]")
        corpus_iter = CorpusIterator(
            file_path=COMPRESSED_SEQ_PATH,
            batch_size=BATCH_ROWS,
            column=SEQ_COL,
            include_ids=tr_ids
        )
        
        w2v_model = Word2Vec(
            sentences=corpus_iter,
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            workers=WORKERS,
            sg=SG,
            epochs=EPOCHS,
            seed=SEED
        )
        
        vocab_size = len(w2v_model.wv)
        print(f"  Vocabulary size: {vocab_size:,}")
        
        # 모델 저장
        model_path = os.path.join(MODEL_DIR, f'w2v_fold{fold}.model')
        w2v_model.save(model_path)
        print(f"  Model saved: {model_path}")
        
        # 검증 세트 임베딩 생성
        print(f"\n[Generating Validation Embeddings]")
        out_valid = os.path.join(OUT_DIR, f'embed_valid_fold{fold}.parquet')
        
        cache_fold_embeddings(
            file_path=COMPRESSED_SEQ_PATH,
            w2v=w2v_model,
            include_ids=va_ids,
            out_path=out_valid,
            batch_size=BATCH_ROWS,
            seq_col=SEQ_COL,
            last_k_list=LAST_K_LIST,
            precision=FLOAT_PRECISION,
            use_float16=USE_FLOAT16
        )
        
        # 메모리 정리
        del w2v_model
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"\n  Fold {fold} completed in {elapsed:.1f} seconds")
    
    # 완료 메시지
    print("\n" + "="*70)
    print("✅ ALL FOLDS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Fold assignment: {FOLD_ASSIGN_PATH}")
    print(f"  - Models: {MODEL_DIR}/w2v_fold*.model")
    print(f"  - Embeddings: {OUT_DIR}/embed_valid_fold*.parquet")
    
    # 컬럼 정보
    n_embedding_cols = VECTOR_SIZE * (3 + len(LAST_K_LIST))
    print(f"\nEmbedding columns:")
    print(f"  - Total: {n_embedding_cols + 2} columns")
    print(f"  - row_id + oov_ratio + {n_embedding_cols} embedding features")
    
    print("\nNext steps:")
    print("  1. Combine OOF embeddings for training")
    print("  2. Generate test embeddings using saved models")
    print("  3. Use embeddings as features for downstream models")


if __name__ == "__main__":
    main()