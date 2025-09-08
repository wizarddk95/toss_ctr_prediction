"""
시퀀스 데이터의 연속 중복 제거 스크립트
- 연속된 동일 토큰을 제거하여 시퀀스 압축
- 메모리 효율적인 배치 처리
- 에러 핸들링 및 진행 상황 표시 개선
"""

import os
import time
import pandas as pd
import pyarrow as pa  # ⚠️ 추가된 임포트
import pyarrow.parquet as pq
from itertools import groupby
from typing import Optional
import warnings

# Swifter는 선택적으로 사용
try:
    import swifter
    USE_SWIFTER = True
except ImportError:
    USE_SWIFTER = False
    warnings.warn("Swifter not installed. Using standard pandas apply.")


def compress_sequence(seq_str: str) -> str:
    """
    한 개의 seq 문자열에서 연속 중복을 제거하는 함수
    
    Example:
        "1,1,2,2,2,3,1,1" -> "1,2,3,1"
    """
    if not isinstance(seq_str, str) or not seq_str:
        return ""  # 빈 문자열이나 NaN의 경우 처리
    
    tokens = seq_str.split(',')
    
    # itertools.groupby를 사용하여 연속된 중복을 효율적으로 제거
    deduplicated_tokens = [key for key, _ in groupby(tokens)]
    
    return ','.join(deduplicated_tokens)


def compress_sequence_vectorized(seq_series: pd.Series) -> pd.Series:
    """
    벡터화된 방식으로 시퀀스 압축 (더 빠를 수 있음)
    """
    def compress_fast(seq_str):
        if pd.isna(seq_str) or seq_str == "":
            return ""
        tokens = seq_str.split(',')
        result = [tokens[0]] if tokens else []
        for token in tokens[1:]:
            if token != result[-1]:
                result.append(token)
        return ','.join(result)
    
    return seq_series.apply(compress_fast)


def get_file_info(file_path: str) -> dict:
    """파일 정보 가져오기"""
    pf = pq.ParquetFile(file_path)
    metadata = pf.metadata
    return {
        'num_rows': metadata.num_rows,
        'num_row_groups': metadata.num_row_groups,
        'size_mb': os.path.getsize(file_path) / (1024 * 1024)
    }


def create_compressed_sequence_file(
    input_path: str, 
    output_path: str, 
    batch_size: int = 200_000,
    use_parallel: bool = True,
    show_stats: bool = True
):
    """
    'seq' 컬럼의 연속 중복을 제거하고 결과를 새로운 Parquet 파일로 저장
    
    Args:
        input_path: 입력 Parquet 파일 경로
        output_path: 출력 Parquet 파일 경로  
        batch_size: 배치 크기
        use_parallel: 병렬 처리 사용 여부 (Swifter 필요)
        show_stats: 통계 정보 표시 여부
    """
    print(f"\n{'='*60}")
    print(f"Compressing sequences: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    
    # 입력 파일 정보
    if show_stats:
        info = get_file_info(input_path)
        print(f"Input file info:")
        print(f"  - Total rows: {info['num_rows']:,}")
        print(f"  - Size: {info['size_mb']:.2f} MB")
        print(f"  - Batch size: {batch_size:,}")
    
    start_time = time.time()
    writer = None
    total_rows_processed = 0
    compression_stats = []
    
    try:
        parquet_file = pq.ParquetFile(input_path)
        total_batches = (info['num_rows'] // batch_size) + 1 if show_stats else "?"
        
        for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=batch_size, columns=['seq'])):
            batch_start = time.time()
            
            # Arrow 배치를 Pandas DataFrame으로 변환
            batch_df = batch.to_pandas()
            batch_rows = len(batch_df)
            
            # 압축 전 평균 길이 계산 (선택적)
            if show_stats and batch_idx == 0:
                avg_len_before = batch_df['seq'].fillna('').str.count(',').mean() + 1
            
            # 압축 수행
            if use_parallel and USE_SWIFTER:
                compressed_series = batch_df['seq'].swifter.apply(compress_sequence)
                method = "Swifter (parallel)"
            else:
                # 기본 apply 또는 벡터화 방식
                compressed_series = compress_sequence_vectorized(batch_df['seq'])
                method = "Vectorized"
            
            # 압축 후 평균 길이 계산 (선택적)
            if show_stats and batch_idx == 0:
                avg_len_after = compressed_series.fillna('').str.count(',').mean() + 1
                compression_ratio = (1 - avg_len_after / avg_len_before) * 100 if avg_len_before > 0 else 0
                compression_stats.append(compression_ratio)
            
            # 결과를 pyarrow 테이블로 변환하여 파일에 쓰기
            result_df = pd.DataFrame({'seq_compressed': compressed_series})
            table = pa.Table.from_pandas(result_df, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)
            
            # 진행 상황 업데이트
            total_rows_processed += batch_rows
            batch_time = time.time() - batch_start
            
            print(f"  Batch {batch_idx+1}/{total_batches} | "
                  f"Rows: {total_rows_processed:,} | "
                  f"Time: {batch_time:.2f}s | "
                  f"Method: {method}", end='\r')
        
        print()  # 줄바꿈
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        raise
    
    finally:
        if writer is not None:
            writer.close()
    
    # 완료 통계
    elapsed_time = time.time() - start_time
    
    if show_stats and os.path.exists(output_path):
        output_info = get_file_info(output_path)
        
        print(f"\n{'='*60}")
        print(f"✅ Compression Complete!")
        print(f"{'='*60}")
        print(f"Output file: {output_path}")
        print(f"  - Total rows: {output_info['num_rows']:,}")
        print(f"  - Size: {output_info['size_mb']:.2f} MB")
        print(f"  - Size reduction: {(1 - output_info['size_mb']/info['size_mb'])*100:.1f}%")
        
        if compression_stats:
            print(f"  - Avg sequence compression: {compression_stats[0]:.1f}%")
        
        print(f"  - Processing time: {elapsed_time:.2f} seconds")
        print(f"  - Speed: {total_rows_processed/elapsed_time:.0f} rows/sec")
    else:
        print(f"\n✅ Done. Processed {total_rows_processed:,} rows in {elapsed_time:.2f} seconds")


def validate_compression(input_path: str, output_path: str, sample_size: int = 5):
    """압축 결과 검증 (샘플 확인)"""
    print(f"\n{'='*60}")
    print("Validation: Checking compression results")
    print(f"{'='*60}")
    
    # 원본과 압축된 데이터 샘플 읽기
    original = pd.read_parquet(input_path, columns=['seq']).head(sample_size)
    compressed = pd.read_parquet(output_path, columns=['seq_compressed']).head(sample_size)
    
    for i in range(min(sample_size, len(original))):
        orig_seq = original.iloc[i]['seq']
        comp_seq = compressed.iloc[i]['seq_compressed']
        
        if pd.notna(orig_seq):
            orig_tokens = orig_seq.split(',')[:10]  # 처음 10개만 표시
            comp_tokens = comp_seq.split(',')[:10]
            
            print(f"\nRow {i}:")
            print(f"  Original  ({len(orig_seq.split(','))} tokens): {','.join(orig_tokens)}...")
            print(f"  Compressed ({len(comp_seq.split(','))} tokens): {','.join(comp_tokens)}...")


# =========================
# 메인 실행 함수
# =========================
def main():
    """메인 실행 함수"""
    
    # 경로 설정
    train_input = './data/train_optimized.parquet'
    train_output = './data/seq_compression/train_seq_compressed.parquet'
    test_input = './data/test_optimized.parquet'
    test_output = './data/seq_compression/test_seq_compressed.parquet'
    
    # 파일 존재 확인
    files_to_process = []
    if os.path.exists(train_input):
        files_to_process.append((train_input, train_output))
    else:
        print(f"⚠️ Train file not found: {train_input}")
    
    if os.path.exists(test_input):
        files_to_process.append((test_input, test_output))
    else:
        print(f"⚠️ Test file not found: {test_input}")
    
    if not files_to_process:
        print("❌ No files to process!")
        return
    
    # 처리 옵션
    use_parallel = USE_SWIFTER  # Swifter 설치 여부에 따라 자동 설정
    batch_size = 200_000
    
    print(f"\n{'='*60}")
    print("SEQUENCE COMPRESSION PIPELINE")
    print(f"{'='*60}")
    print(f"Parallel processing: {'Enabled (Swifter)' if use_parallel else 'Disabled'}")
    print(f"Batch size: {batch_size:,}")
    
    # 각 파일 처리
    for input_path, output_path in files_to_process:
        create_compressed_sequence_file(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            use_parallel=use_parallel,
            show_stats=True
        )
        
        # 첫 번째 파일만 검증 (옵션)
        if input_path == files_to_process[0][0]:
            validate_compression(input_path, output_path, sample_size=3)
    
    print(f"\n{'='*60}")
    print("ALL TASKS COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()