import numpy as np
from sklearn.metrics import average_precision_score

def weighted_log_loss(y_true, y_pred):
    # 클래스 가중치를 50:50으로 맞춤
    pos_weight = 0.5 / np.mean(y_true)
    neg_weight = 0.5 / (1 - np.mean(y_true))
    loss = - (pos_weight * y_true * np.log(y_pred + 1e-15) + neg_weight * (1 - y_true) * np.log(1 - y_pred + 1e-15))
    return np.mean(loss)

def toss_metric(y_true, y_pred):
    ap = average_precision_score(y_true, y_pred)
    wll = weighted_log_loss(y_true, y_pred)
    score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    return score

# LightGBM용 커스텀 eval 함수
def lgbm_toss_metric(y_true, y_pred):
    """is_higher_better=True 이므로 score를 그대로 반환"""
    return 'toss_score', toss_metric(y_true, y_pred), True