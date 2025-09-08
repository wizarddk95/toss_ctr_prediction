import os
import math
import gc
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
FILE_PATHS = ['./data/train_optimized.parquet']
SEQ_COL = 'seq'
LABEL_COL = 'clicked'
USE_STRATIFY = True
BATCH_ROWS = 200_000
VECTOR_SIZE = 64
WINDOW = 5
MIN_COUNT = 1
WORKERS = multiprocessing.cpu_count()
SG = 1 # 1: skip-gram, 0: CBOW
EPOCHS = 3
LAST_K_LIST = (5, 20, 50)
N_SPLITS = 5
SEED = 42

# 저장 정밀도 설정 (소수점 자릿수)
FLOAT_PRECISION = 4  # 소수점 4자리까지만 저장
USE_FLOAT16 = True  # True로 설정시 float16 사용 (더 작은 크기, 낮은 정밀도)

OUT_DIR = './data/seq_w2v_embedding'
FOLD_ASSIGN_PATH = os.path.join(OUT_DIR, 'fold_assign.parquet')
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# 유틸: 말뭉치 이터레이터(선택 row_id만 포함)
# =========================
class CorpusIterator:
    """
    Parquet 파일을 배치 단위로 읽어 메모리 효율적으로 문장(토큰 리스트) 스트리밍.
    include_ids가 주어지면 해당 row_id만 포함.
    """
    def __init__(self,
                 file_paths: List[str],
                 batch_size: int = 100_000,
                 column: str = 'seq',
                 include_ids: Optional[Set[int]] = None):
        self.file_paths = file_paths
        self.batch_size = batch_size
        self.column = column
        self.include_ids = include_ids

    def __iter__(self):
        global_row = 0
        for file_path in self.file_paths:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, columns=[self.column]):
                series = batch.to_pandas()[self.column]
                for s in series:
                    take = (self.include_ids is None) or (global_row in self.include_ids)
                    if take:
                        if isinstance(s, str) and s:
                            # 빠른 분할(정수 ID면 np.fromstring 권장)
                            toks = [t for t in s.split(',') if t]
                            yield toks
                        else:
                            yield []
                    global_row += 1


# =========================
# 유틸: row_id와 label 수집(스트리밍)
# =========================
def count_total_rows(file_paths: List[str], batch_size: int, col: str) -> int:
    total = 0
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[col]):
            total += len(batch.to_pandas())
    return total    

def collect_labels(file_paths: List[str],
                   label_col: str,
                   batch_size: int = 200_000) -> pd.DataFrame:
    """
    라벨을 메모리 효율적으로 수집하여 (row_id, label) DataFrame 생성.
    """
    rows = []
    global_row = 0
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[label_col]):
            ser = batch.to_pandas()[label_col]
            cnt = len(ser)
            rows.append(pd.DataFrame({
                'row_id': np.arange(global_row, global_row + cnt, dtype=np.int64),
                'label': ser.values
            }))
            global_row += cnt
    meta = pd.concat(rows, ignore_index=True)
    return meta


def load_or_make_folds() -> pd.DataFrame:
    """
    fold_assign.parquet가 있으면 로드, 없으면 생성하여 저장.
    반환: DataFrame[row_id, fold]  (row_id = 0..N-1)
    """
    if os.path.exists(FOLD_ASSIGN_PATH):
        print(f"[fold] Reusing existing fold assignment: {FOLD_ASSIGN_PATH}")
        assign = pd.read_parquet(FOLD_ASSIGN_PATH)
        # 검증
        assert {'row_id', 'fold'} <= set(assign.columns)
        return assign[['row_id', 'fold']].copy()

    print("[fold] Creating new fold assignment...")
    if USE_STRATIFY:
        meta = collect_labels(FILE_PATHS, LABEL_COL, BATCH_ROWS)
        y = meta['label'].values
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(skf.split(meta['row_id'].values, y))
        assign = pd.DataFrame({'row_id': meta['row_id'].values, 'fold': -1}, dtype=np.int64)
        for f, (_, va_idx) in enumerate(splits):
            assign.loc[va_idx, 'fold'] = f
    else:
        total = count_total_rows(FILE_PATHS, BATCH_ROWS, SEQ_COL)
        meta = pd.DataFrame({'row_id': np.arange(total, dtype=np.int64)})
        kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        splits = list(kf.split(meta['row_id'].values))
        assign = pd.DataFrame({'row_id': meta['row_id'].values, 'fold': -1}, dtype=np.int64)
        for f, (_, va_idx) in enumerate(splits):
            assign.loc[va_idx, 'fold'] = f

    assign.to_parquet(FOLD_ASSIGN_PATH, index=False)
    print(f"[fold] Saved: {FOLD_ASSIGN_PATH}")
    return assign


# =========================
# 유틸: 정밀도 감소 함수
# =========================
def reduce_precision(arr: np.ndarray, precision: int = 4, use_float16: bool = False) -> np.ndarray:
    """
    배열의 정밀도를 줄여 저장 공간 절약
    
    Args:
        arr: 입력 배열
        precision: 소수점 자릿수 (반올림)
        use_float16: float16 타입 사용 여부
    """
    # 소수점 자릿수 제한
    arr_rounded = np.round(arr, precision)
    
    # float16으로 변환 (옵션)
    if use_float16:
        return arr_rounded.astype(np.float16)
    else:
        return arr_rounded.astype(np.float32)


# =========================
# 유틸: 대표 임베딩 계산 (메타피처 제거)
# =========================
def sequence_vectors(tokens: List[str],
                     w2v: Word2Vec,
                     last_k_list: Iterable[int]) -> Dict[str, np.ndarray]:
    """
    한 시퀀스에서 여러 대표 임베딩( mean, last, last-k들, max )을 계산.
    사전에 없는 토큰은 건너뜀. 모두 없으면 zero-vector.
    """
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    d = w2v.vector_size
    if len(vecs) == 0:
        zero = np.zeros(d, dtype=np.float32)
        out = {'mean': zero, 'last': zero, 'max': zero}
        for k in last_k_list:
            out[f'last{k}'] = zero
        return out

    V = np.vstack(vecs).astype(np.float32)

    # mean
    mean_vec = V.mean(axis=0)

    # last
    last_vec = V[-1]

    # last-k
    out = {'mean': mean_vec, 'last': last_vec}
    for k in last_k_list:
        if len(V) >= k:
            out[f'last{k}'] = V[-k:].mean(axis=0)
        else:
            out[f'last{k}'] = V.mean(axis=0)

    # max pooling (각 차원별 최대값)
    max_vec = V.max(axis=0)
    out['max'] = max_vec

    return out


def expand_vec(prefix: str, v: np.ndarray, precision: int = 4) -> Dict[str, float]:
    """
    벡터를 {f'{prefix}_{i}': value} 딕셔너리로 펼침.
    정밀도를 제한하여 저장 공간 절약.
    """
    # 정밀도 제한
    v_reduced = np.round(v, precision)
    return {f'{prefix}_{i}': float(v_reduced[i]) for i in range(v.shape[0])}


# =========================
# 개선된 대표 임베딩 계산 로직 (메타피처 제거, 정밀도 감소)
# =========================
def process_row_to_embeddings(tokens: List[str], 
                             w2v: Word2Vec, 
                             last_k_list: Iterable[int],
                             precision: int = 4,
                             use_float16: bool = False) -> Dict:
    """
    Apply 함수에 사용될 워커 함수.
    한 행(토큰 리스트)을 받아 모든 대표 임베딩을 dict로 반환합니다.
    메타피처는 제외하고, 정밀도를 감소시킵니다.
    """
    # 1. 벡터 변환
    vecs = [w2v.wv[t] for t in tokens if t in w2v.wv]
    d = w2v.vector_size
    
    # 2. 대표 임베딩 생성
    if not vecs:
        zero = np.zeros(d, dtype=np.float32)
        reps = {'mean': zero, 'last': zero, 'max': zero}
        for k in last_k_list:
            reps[f'last{k}'] = zero
    else:
        V = np.vstack(vecs).astype(np.float32)
        reps = {'mean': V.mean(axis=0), 'last': V[-1], 'max': V.max(axis=0)}
        for k in last_k_list:
            reps[f'last{k}'] = V[-k:].mean(axis=0) if len(V) >= k else V.mean(axis=0)

    # 3. 정밀도 감소 적용
    for name in reps:
        reps[name] = reduce_precision(reps[name], precision, use_float16)
    
    # 4. 펼친 임베딩을 최종 레코드에 통합 (메타피처 제외)
    rec = {}
    for name, vec in reps.items():
        rec.update(expand_vec(name, vec, precision=0))  # 이미 감소된 정밀도이므로 추가 반올림 불필요
        
    return rec


def cache_fold_embeddings_vectorized(file_paths: List[str],
                                     w2v: Word2Vec,
                                     include_ids: Set[int],
                                     out_path: str,
                                     batch_size: int = 200_000,
                                     seq_col: str = 'seq',
                                     last_k_list: Iterable[int] = (5, 20),
                                     precision: int = 4,
                                     use_float16: bool = False):
    """
    Vectorized version of cache_fold_embeddings with reduced precision.
    """
    writer = None
    all_row_ids = np.array(sorted(list(include_ids)))
    
    row_cursor = 0
    for path in file_paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=[seq_col]):
            batch_start_row = row_cursor
            batch_end_row = row_cursor + len(batch)
            
            # 현재 배치에 포함된 대상 row_id들을 찾음
            target_mask = (all_row_ids >= batch_start_row) & (all_row_ids < batch_end_row)
            target_ids_in_batch = all_row_ids[target_mask]

            if len(target_ids_in_batch) > 0:
                ser = batch.to_pandas()[seq_col]
                
                # 배치 내에서 대상이 되는 Series만 필터링
                local_indices = target_ids_in_batch - batch_start_row
                target_ser = ser.iloc[local_indices]
                
                # 토큰화 (Vectorized)
                tokens_series = target_ser.fillna('').str.split(',')
                
                # Apply 함수를 사용하여 한 번에 모든 임베딩 계산 (Vectorized)
                embedding_records = tokens_series.apply(
                    lambda tokens: process_row_to_embeddings(
                        tokens, w2v, last_k_list, precision, use_float16
                    )
                )
                
                # 결과를 DataFrame으로 변환
                df = pd.DataFrame.from_records(embedding_records.tolist(), index=target_ser.index)
                df['row_id'] = target_ids_in_batch
                
                # 데이터 타입 최적화
                if use_float16:
                    # float 컬럼들을 float16으로 변환
                    float_cols = [col for col in df.columns if col != 'row_id']
                    for col in float_cols:
                        df[col] = df[col].astype(np.float16)

                # 파일에 쓰기
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema, compression='snappy')
                writer.write_table(table)

            row_cursor = batch_end_row

    if writer is not None:
        writer.close()


# =========================
# 메인: 배정 로드→폴드 루프(W2V 학습→임베딩 캐시)
# =========================
def main():
    assign = load_or_make_folds()
    total_rows = assign.shape[0]
    print(f"[main] total_rows={total_rows:,}, n_splits={N_SPLITS}")
    print(f"[config] FLOAT_PRECISION={FLOAT_PRECISION}, USE_FLOAT16={USE_FLOAT16}")
    
    # 예상 크기 감소 계산
    original_bits = 32  # float32
    reduced_bits = 16 if USE_FLOAT16 else 32
    size_reduction = (1 - reduced_bits/original_bits) * 100 if USE_FLOAT16 else 0
    
    if USE_FLOAT16:
        print(f"[info] Using float16 will reduce storage by approximately {size_reduction:.1f}%")
    else:
        print(f"[info] Using float32 with {FLOAT_PRECISION} decimal precision")

    for fold in range(N_SPLITS):
        va_ids = set(assign.loc[assign['fold'] == fold, 'row_id'].tolist())
        tr_ids = set(assign.loc[assign['fold'] != fold, 'row_id'].tolist())
        print(f"\n===== FOLD {fold} =====")
        print(f"train={len(tr_ids):,}  valid={len(va_ids):,}")

        # 폴드 학습 말뭉치
        corpus_iter = CorpusIterator(
            file_paths=FILE_PATHS,
            batch_size=BATCH_ROWS,
            column=SEQ_COL,
            include_ids=tr_ids
        )

        # 폴드별 Word2Vec 학습
        print("[w2v] training...")
        w2v_model = Word2Vec(
            sentences=corpus_iter,
            vector_size=VECTOR_SIZE,
            window=WINDOW,
            min_count=MIN_COUNT,
            workers=WORKERS,
            sg=SG,
            epochs=EPOCHS
        )
        print("[w2v] done.")

        # 모델 저장(선택)
        w2v_path = os.path.join('./models', f'w2v_fold{fold}.model')
        w2v_model.save(w2v_path)

        # 검증 임베딩 캐시 (OOF)
        out_va = os.path.join(OUT_DIR, f'embed_valid_fold{fold}.parquet')
        print(f"[cache] valid embeddings -> {out_va}")
        cache_fold_embeddings_vectorized(
            file_paths=FILE_PATHS,
            w2v=w2v_model,
            include_ids=va_ids,
            out_path=out_va,
            batch_size=BATCH_ROWS,
            seq_col=SEQ_COL,
            last_k_list=LAST_K_LIST,
            precision=FLOAT_PRECISION,
            use_float16=USE_FLOAT16
        )
        
        del w2v_model
        gc.collect()

    print("\n[done] All folds processed.")
    print(f"Fold assignment: {FOLD_ASSIGN_PATH}")
    print(f"Embeddings dir : {os.path.abspath(OUT_DIR)}")
    
    # 최종 정보 출력
    print("\n[Storage Info]")
    print(f"- Precision: {FLOAT_PRECISION} decimal places")
    print(f"- Data type: {'float16' if USE_FLOAT16 else 'float32'}")
    print(f"- Columns per file: {1 + VECTOR_SIZE * (3 + len(LAST_K_LIST))} (row_id + embeddings)")
    print(f"- Meta features: REMOVED (seq_len, seq_unique_len excluded)")


if __name__ == "__main__":
    main()