import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from src.metrics import toss_metric, lgbm_toss_metric


"""
저장된 Fold 인덱스를 활용한 LightGBM 학습
Word2Vec 학습 시 사용한 동일한 fold를 사용하여 재현성 확보
"""

# ===================================
# 1. Fold 인덱스 로드
# ===================================

def load_fold_assignments(fold_path: str = './data/seq_w2v_embedding/fold_assign.parquet'):
    """
    Word2Vec 학습 시 저장한 fold 할당 로드
    
    Returns:
        DataFrame with columns: row_id, fold
    """
    if not os.path.exists(fold_path):
        print(f"⚠️ Fold assignment file not found: {fold_path}")
        print("Using new random splits instead...")
        return None
    
    fold_assign = pd.read_parquet(fold_path)
    print(f"✅ Loaded fold assignments: {fold_path}")
    print(f"   Shape: {fold_assign.shape}")
    print(f"   Folds: {fold_assign['fold'].unique()}")
    
    return fold_assign


def get_fold_indices(fold_assign: pd.DataFrame, fold_num: int):
    """
    특정 fold의 train/validation 인덱스 반환
    
    Args:
        fold_assign: fold 할당 DataFrame
        fold_num: fold 번호 (0-4)
    
    Returns:
        train_idx, val_idx
    """
    val_idx = fold_assign[fold_assign['fold'] == fold_num]['row_id'].values
    train_idx = fold_assign[fold_assign['fold'] != fold_num]['row_id'].values
    
    return train_idx, val_idx


# ===================================
# 2. 메인 학습 코드 (수정 버전)
# ===================================

def train_with_saved_folds(train_df, test_df, features, categorical_features, 
                          fold_assign_path='./data/seq_w2v_embedding/fold_assign.parquet'):
    """
    저장된 fold를 사용한 LightGBM 학습
    """
    
    TARGET = 'clicked'
    N_SPLITS = 5
    
    # 카테고리 피처 타입 변경
    for col in categorical_features:
        if col in train_df.columns:
            if train_df[col].dtype == 'float16':
                train_df[col] = train_df[col].astype('float32')
            train_df[col] = train_df[col].astype('category')
        if col in test_df.columns:
            if test_df[col].dtype == 'float16':
                test_df[col] = test_df[col].astype('float32')
            test_df[col] = test_df[col].astype('category')
    
    print("Feature types updated.")
    
    X_train = train_df[features]
    y_train = train_df[TARGET]
    
    # Fold 할당 로드
    fold_assign = load_fold_assignments(fold_assign_path)
    
    if fold_assign is None:
        # Fallback: 새로운 fold 생성
        print("Creating new fold splits...")
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        fold_iterator = enumerate(skf.split(X_train, y_train))
    else:
        # 저장된 fold 사용
        print("Using saved fold assignments for reproducibility")
        
        # train_df의 순서와 fold_assign의 row_id가 일치하는지 확인
        assert len(train_df) == len(fold_assign), \
            f"Size mismatch: train_df({len(train_df)}) != fold_assign({len(fold_assign)})"
        
        # fold_assign이 row_id 순서대로 정렬되어 있는지 확인
        fold_assign = fold_assign.sort_values('row_id').reset_index(drop=True)
        
        # Custom fold iterator
        fold_iterator = []
        for fold_num in range(N_SPLITS):
            train_idx, val_idx = get_fold_indices(fold_assign, fold_num)
            fold_iterator.append((fold_num, (train_idx, val_idx)))
    
    # 학습 시작
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    cv_scores = []
    feature_importance_list = []
    
    for fold, (train_idx, val_idx) in fold_iterator:
        print(f"\n===== Fold {fold} =====")
        print(f"Train size: {len(train_idx):,}, Val size: {len(val_idx):,}")
        
        # 데이터 분할
        X_train_fold = X_train.iloc[train_idx]
        y_train_fold = y_train.iloc[train_idx]
        X_val_fold = X_train.iloc[val_idx]
        y_val_fold = y_train.iloc[val_idx]
        
        # 클래스 불균형 해소
        scale_pos_weight = np.sum(y_train_fold == 0) / np.sum(y_train_fold == 1)
        print(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # LightGBM 모델
        lgbm = lgb.LGBMClassifier(
            objective='binary',
            metric='none',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            sub_sample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1
        )
        
        # 학습
        lgbm.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            eval_metric=lgbm_toss_metric,  # 사용자 정의 메트릭
            callbacks=[lgb.early_stopping(100, verbose=True)]
        )
        
        # 예측
        val_preds = lgbm.predict_proba(X_val_fold)[:, 1]
        oof_preds[val_idx] = val_preds
        
        # 평가
        fold_score = toss_metric(y_val_fold, val_preds)
        cv_scores.append(fold_score)
        print(f"Fold {fold} Score: {fold_score:.5f}")
        
        # Feature importance 저장
        importance = pd.DataFrame({
            'feature': features,
            'importance': lgbm.feature_importances_,
            'fold': fold
        })
        feature_importance_list.append(importance)
        
        # 테스트 예측 (앙상블)
        test_preds += lgbm.predict_proba(test_df[features])[:, 1] / N_SPLITS
        
        # 메모리 정리
        del lgbm, X_train_fold, X_val_fold, y_train_fold, y_val_fold
        import gc
        gc.collect()
    
    # 최종 결과
    print(f"\n" + "="*60)
    print(f"Average CV Score: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")
    print("="*60)
    
    # Feature importance 집계
    feature_importance = pd.concat(feature_importance_list, ignore_index=True)
    feature_importance_mean = feature_importance.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print("\nTop 10 Important Features:")
    for feat, imp in feature_importance_mean.head(10).items():
        print(f"  {feat}: {imp:.1f}")
    
    return oof_preds, test_preds, cv_scores, feature_importance


# ===================================
# 3. 실행 및 저장
# ===================================

def main(train_df, test_df):
    """메인 실행 함수"""
    
    # 설정
    TARGET = 'clicked'
    features = [col for col in train_df.columns if col not in ['ID', 'seq', TARGET]]
    
    categorical_features = [
        'gender', 'age_group', 'inventory_id', 'day_of_week', 'hour',
        'seq_len', 'seq_first', 'seq_last', 'seq_max', 'seq_min', 
        'seq_mean', 'seq_std', 'l_feat_14', 'age_inv_interaction',
        'is_last_in_streak'
    ]
    
    # 학습 (저장된 fold 사용)
    oof_preds, test_preds, cv_scores, feature_importance = train_with_saved_folds(
        train_df=train_df,
        test_df=test_df,
        features=features,
        categorical_features=categorical_features,
        fold_assign_path='./data/seq_w2v_embedding/fold_assign.parquet'
    )
    
    # 타임스탬프
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. 제출 파일 생성
    submission = pd.read_csv('./data/raw_data/sample_submission.csv')
    submission['clicked'] = test_preds
    submission.to_csv(f'./submissions/submission_{now}.csv', index=False)
    print(f"\n✅ Submission saved: submission_{now}.csv")
    
    # 2. OOF 예측 저장 (fold 정보 포함)
    fold_assign = pd.read_parquet('./data/seq_w2v_embedding/fold_assign.parquet')
    oof_df = pd.DataFrame({
        'row_id': fold_assign['row_id'],
        'fold': fold_assign['fold'],
        'y_true': train_df[TARGET].values,
        'oof_pred': oof_preds
    })
    oof_df.to_csv(f'./oof_preds/oof_preds_{now}.csv', index=False)
    print(f"✅ OOF predictions saved: oof_preds_{now}.csv")
    
    # 3. CV 점수 저장
    cv_results = pd.DataFrame({
        'fold': range(len(cv_scores)),
        'score': cv_scores,
        'mean': np.mean(cv_scores),
        'std': np.std(cv_scores)
    })
    cv_results.to_csv(f'./cv_results/cv_results_{now}.csv', index=False)
    print(f"✅ CV results saved: cv_results_{now}.csv")
    
    # 4. Feature importance 저장
    feature_importance.to_csv(f'./feature_importance/importance_{now}.csv', index=False)
    print(f"✅ Feature importance saved: importance_{now}.csv")
    
    return oof_preds, test_preds
