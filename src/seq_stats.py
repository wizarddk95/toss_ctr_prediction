import numpy as np
import pandas as pd
import swifter

def extract_stats(seq_str: str):
    if not seq_str:  # 빈 문자열이면 바로 리턴
        return (0, 0, 0, 0, 0, 0, 0, 0)

    # "12,45,89" 같은 문자열을 numpy array로 변환
    arr = np.fromstring(seq_str, sep=",", dtype=np.int16)

    if arr.size == 0:  # 변환 실패 or 빈값 → 안전하게 리턴
        return (0, 0, 0, 0, 0, 0, 0, 0)

    # arr_diff = np.diff(arr) if arr.size > 1 else np.array([0])

    # 여기서 각 통계량 계산
    return (
        len(arr),                 # seq_len
        len(np.unique(arr)),      # seq_nunique
        arr[0],                   # seq_first
        arr[-1],                  # seq_last
        arr.max(),                # seq_max
        arr.min(),                # seq_min
        arr.mean(),               # seq_mean
        arr.std(),                # seq_std
        # arr.sum(),                # seq_sum
        # arr_diff.mean(),          # diff_mean
        # arr_diff.std(),           # diff_std
        # arr_diff.min(),           # diff_min
        # arr_diff.max()            # diff_max
    )

def generate_seq_stats(df, col="seq"):
    print("Generating stats features with swifter...")
    stats = df[col].swifter.apply(extract_stats)
    stats_df = pd.DataFrame(stats.tolist(), columns=["seq_len","seq_unique_len","seq_first","seq_last","seq_max","seq_min","seq_mean","seq_std"])
    return pd.concat([df, stats_df], axis=1)

if __name__ == "__main__":
    train_seq_df = pd.read_parquet('./data/train_optimized.parquet', columns=['seq'])
    test_seq_df = pd.read_parquet('./data/test_optimized.parquet', columns=['seq'])

    train_seq_stats = generate_seq_stats(train_seq_df, col="seq")
    test_seq_stats = generate_seq_stats(test_seq_df, col="seq")

    train_seq_stats.to_parquet('./data/seq_stats/train_seq_stats.parquet')
    test_seq_stats.to_parquet('./data/seq_stats/test_seq_stats.parquet')

    print("Sequence stats generated.")
