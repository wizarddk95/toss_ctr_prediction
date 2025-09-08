"""
시퀀스 반복 패턴 특징 추출 스크립트
- 연속 중복 패턴 분석
- 최대 연속 길이, 중복 비율 등 계산
- 메모리 효율적인 배치 처리
"""

import os
import re
import time
import gc
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings

# Swifter는 선택적으로 사용
try:
    import swifter
    USE_SWIFTER = True
except ImportError:
    USE_SWIFTER = False
    warnings.warn("Swifter not installed. Using standard pandas apply.")


def get_repetition_features(seq_str: str) -> Dict[str, float]:
    """
    시퀀스의 반복 패턴 특징을 추출
    
    Returns:
        dict: 다음 특징들을 포함
            - max_streak: 최대 연속 반복 길이
            - consecutive_dupe_ratio: 전체 토큰 중 연속 중복 비율
            - is_last_in_streak: 마지막 토큰이 연속 중복인지 여부
            - num_unique_streaks: 고유한 연속 패턴 개수
            - avg_streak_length: 평균 연속 길이
    """
    if not isinstance(seq_str, str) or not seq_str:
        return {
            "max_streak": 0,
            "consecutive_dupe_ratio": 0.0,
            "is_last_in_streak": 0,
            "num_unique_streaks": 0,
            "avg_streak_length": 0.0
        }
    
    tokens = seq_str.split(',')
    n_tokens = len(tokens)
    
    if n_tokens <= 1:
        return {
            "max_streak": 1,
            "consecutive_dupe_ratio": 0.0,
            "is_last_in_streak": 0,
            "num_unique_streaks": 0,
            "avg_streak_length": 1.0
        }
    
    # 연속 중복 찾기 (더 효율적인 방법)
    streaks = []
    current_streak = 1
    streak_tokens = []
    
    for i in range(1, n_tokens):
        if tokens[i] == tokens[i-1]:
            current_streak += 1
        else:
            if current_streak > 1:
                streaks.append(current_streak)
                streak_tokens.append(tokens[i-1])
            current_streak = 1
    
    # 마지막 스트릭 처리
    if current_streak > 1:
        streaks.append(current_streak)
        streak_tokens.append(tokens[-1])
    
    # 특징 계산
    if streaks:
        max_streak = max(streaks)
        num_consecutive_dupes = sum(s - 1 for s in streaks)  # 중복된 토큰 수
        num_unique_streaks = len(set(streak_tokens))
        avg_streak_length = np.mean(streaks)
    else:
        max_streak = 1
        num_consecutive_dupes = 0
        num_unique_streaks = 0
        avg_streak_length = 0.0
    
    # 마지막 토큰이 연속 중복인지 확인
    is_last_in_streak = 1 if tokens[-1] == tokens[-2] else 0
    
    # 비율 계산
    consecutive_dupe_ratio = num_consecutive_dupes / n_tokens
    
    return {
        "max_streak": max_streak,
        "consecutive_dupe_ratio": consecutive_dupe_ratio,
        "is_last_in_streak": is_last_in_streak,
        "num_unique_streaks": num_unique_streaks,
        "avg_streak_length": avg_streak_length
    }


def get_repetition_features_regex(seq_str: str) -> Dict[str, float]:
    """
    정규표현식을 사용한 대체 구현 (비교용)
    """
    if not isinstance(seq_str, str) or not seq_str:
        return {
            "max_streak": 0,
            "consecutive_dupe_ratio": 0.0,
            "is_last_in_streak": 0
        }
    
    tokens = seq_str.split(',')
    n_tokens = len(tokens)
    
    if n_tokens <= 1:
        return {
            "max_streak": 1,
            "consecutive_dupe_ratio": 0.0,
            "is_last_in_streak": 0
        }
    
    # 연속된 중복을 찾기 위한 개선된 정규표현식
    # 토큰별로 처리하는 것이 더 정확
    streaks = []
    i = 0
    while i < n_tokens:
        current = tokens[i]
        count = 1
        while i + count < n_tokens and tokens[i + count] == current:
            count += 1
        if count > 1:
            streaks.append(count)
        i += count
    
    if streaks:
        max_streak = max(streaks)
        num_consecutive_dupes = sum(s - 1 for s in streaks)
    else:
        max_streak = 1
        num_consecutive_dupes = 0
    
    is_last_in_streak = 1 if tokens[-1] == tokens[-2] else 0
    consecutive_dupe_ratio = num_consecutive_dupes / n_tokens
    
    return {
        "max_streak": max_streak,
        "consecutive_dupe_ratio": consecutive_dupe_ratio,
        "is_last_in_streak": is_last_in_streak
    }


def process_batch_features(df_batch: pd.DataFrame, use_parallel: bool = True) -> pd.DataFrame:
    """
    배치 단위로 특징 추출
    """
    if use_parallel and USE_SWIFTER:
        # Swifter를 사용한 병렬 처리
        features_list = df_batch['seq'].swifter.apply(get_repetition_features).tolist()
    else:
        # 일반 apply
        features_list = df_batch['seq'].apply(get_repetition_features).tolist()
    
    # DataFrame으로 변환 (훨씬 빠름)
    features_df = pd.DataFrame(features_list)
    
    return features_df


def extract_repetition_features(
    input_path: str,
    output_path: str,
    batch_size: int = 200_000,
    use_parallel: bool = True,
    show_stats: bool = True
):
    """
    Parquet 파일에서 시퀀스 반복 패턴 특징을 추출하여 저장
    
    Args:
        input_path: 입력 Parquet 파일 경로
        output_path: 출력 Parquet 파일 경로
        batch_size: 배치 크기
        use_parallel: 병렬 처리 사용 여부
        show_stats: 통계 표시 여부
    """
    print(f"\n{'='*60}")
    print(f"Extracting repetition features: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    start_time = time.time()
    writer = None
    total_rows = 0
    
    # 통계 수집용
    feature_stats = {
        'max_streak': [],
        'consecutive_dupe_ratio': [],
        'is_last_in_streak': [],
        'num_unique_streaks': [],
        'avg_streak_length': []
    }
    
    try:
        pf = pq.ParquetFile(input_path)
        
        if show_stats:
            metadata = pf.metadata
            total_rows_expected = metadata.num_rows
            print(f"Total rows to process: {total_rows_expected:,}")
        
        for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=['seq'])):
            batch_start = time.time()
            
            # Arrow 배치를 Pandas로 변환
            df_batch = batch.to_pandas()
            batch_rows = len(df_batch)
            
            # 특징 추출
            features_df = process_batch_features(df_batch, use_parallel)
            
            # 첫 번째 배치에서 통계 수집 (선택적)
            if show_stats and batch_idx == 0:
                for col in feature_stats.keys():
                    if col in features_df.columns:
                        feature_stats[col] = [
                            features_df[col].mean(),
                            features_df[col].std(),
                            features_df[col].min(),
                            features_df[col].max()
                        ]
            
            # PyArrow 테이블로 변환 및 저장
            table = pa.Table.from_pandas(features_df, preserve_index=False)
            
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
            writer.write_table(table)
            
            # 진행 상황 업데이트
            total_rows += batch_rows
            batch_time = time.time() - batch_start
            
            if show_stats:
                progress = (total_rows / total_rows_expected) * 100
                print(f"  Batch {batch_idx+1} | Progress: {progress:.1f}% | "
                      f"Rows: {total_rows:,} | Time: {batch_time:.2f}s", end='\r')
            
            # 메모리 정리
            del df_batch, features_df, table
            if batch_idx % 10 == 0:
                gc.collect()
        
        print()  # 줄바꿈
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
        
    finally:
        if writer is not None:
            writer.close()
    
    elapsed_time = time.time() - start_time
    
    # 완료 통계
    print(f"\n{'='*60}")
    print(f"✅ Feature extraction complete!")
    print(f"{'='*60}")
    print(f"Output file: {output_path}")
    print(f"  - Total rows: {total_rows:,}")
    print(f"  - Processing time: {elapsed_time:.2f} seconds")
    print(f"  - Speed: {total_rows/elapsed_time:.0f} rows/sec")
    
    if show_stats and feature_stats['max_streak']:
        print(f"\nFeature Statistics (from first batch):")
        for feature, stats in feature_stats.items():
            if stats:
                print(f"  {feature}:")
                print(f"    - Mean: {stats[0]:.3f}")
                print(f"    - Std:  {stats[1]:.3f}")
                print(f"    - Min:  {stats[2]:.3f}")
                print(f"    - Max:  {stats[3]:.3f}")


def validate_features(input_path: str, output_path: str, sample_size: int = 5):
    """
    추출된 특징 검증
    """
    print(f"\n{'='*60}")
    print("Validation: Checking extracted features")
    print(f"{'='*60}")
    
    # 원본과 특징 데이터 샘플 읽기
    original = pd.read_parquet(input_path, columns=['seq']).head(sample_size)
    features = pd.read_parquet(output_path).head(sample_size)
    
    for i in range(min(sample_size, len(original))):
        seq = original.iloc[i]['seq']
        
        if pd.notna(seq):
            tokens = seq.split(',')[:20]  # 처음 20개만 표시
            
            print(f"\nRow {i}:")
            print(f"  Sequence (first 20): {','.join(tokens)}...")
            print(f"  Features:")
            for col in features.columns:
                print(f"    - {col}: {features.iloc[i][col]:.3f}")
            
            # 수동 검증 (선택적)
            manual_features = get_repetition_features(seq)
            print(f"  Manual check:")
            for key, value in manual_features.items():
                if key in features.columns:
                    extracted = features.iloc[i][key]
                    match = "✓" if abs(extracted - value) < 0.001 else "✗"
                    print(f"    - {key}: {value:.3f} {match}")


def main():
    """메인 실행 함수"""
    
    # 파일 경로 설정
    train_input = './data/train_optimized.parquet'
    train_output = './data/seq_repeat_pattern/train_seq_repeat_pattern.parquet'
    test_input = './data/test_optimized.parquet'
    test_output = './data/seq_repeat_pattern/test_seq_repeat_pattern.parquet'
    
    # 처리할 파일 목록
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
    
    # 설정
    batch_size = 200_000
    use_parallel = USE_SWIFTER
    
    print(f"\n{'='*60}")
    print("SEQUENCE REPETITION PATTERN EXTRACTION")
    print(f"{'='*60}")
    print(f"Parallel processing: {'Enabled (Swifter)' if use_parallel else 'Disabled'}")
    print(f"Batch size: {batch_size:,}")
    print(f"Features to extract: max_streak, consecutive_dupe_ratio, is_last_in_streak,")
    print(f"                    num_unique_streaks, avg_streak_length")
    
    # 각 파일 처리
    for input_path, output_path in files_to_process:
        extract_repetition_features(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size,
            use_parallel=use_parallel,
            show_stats=True
        )
        
        # 첫 번째 파일만 검증 (옵션)
        if input_path == files_to_process[0][0]:
            validate_features(input_path, output_path, sample_size=3)
    
    print(f"\n{'='*60}")
    print("✅ ALL SEQUENCE REPEAT PATTERNS GENERATED SUCCESSFULLY!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()