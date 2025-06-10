# main-k-xgb.py

import json
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

from data_and_preprocess import (
    load_train_data, load_test_data,
    compute_sample_weights, Preprocessor
)

# 可调参数：Top K 特征数量
TOP_K = 650
# 文件路径
TRAIN_FILE   = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE    = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE  = "results_xgb_topk.jsonl"
RANDOM_STATE = 42

def train_and_predict_xgb(K=TOP_K, random_state=RANDOM_STATE):
    # 1. 加载数据
    X_avg, y_avg, X_raw, y_raw = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)

    # 2. 样本权重
    w = compute_sample_weights(y_raw)

    # 3. 预处理
    pre = Preprocessor(sparse_thresh=0.05, var_thresh=1e-6)
    X_scaled      = pre.fit_transform(X_avg)
    X_test_scaled = pre.transform(X_test)

    # 4. 特征筛选：ExtraTrees 取 Top K
    from sklearn.ensemble import ExtraTreesRegressor
    et = ExtraTreesRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    et.fit(X_scaled, y_avg)
    importances = et.feature_importances_
    topk_idx = np.argsort(importances)[::-1][:K]
    X_scaled      = X_scaled[:, topk_idx]
    X_test_scaled = X_test_scaled[:, topk_idx]

    # 5. 划分训练/验证，并归一化权重
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_scaled, y_avg, w, test_size=0.1, random_state=random_state
    )
    w_tr  = w_tr  / np.mean(w_tr)

    # 6. XGBoost 训练（固定 n_estimators，不做早停）
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        learning_rate=0.01,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        tree_method='hist',
        verbosity=1,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr)

    # 7. 预测并写结果
    preds = model.predict(X_test_scaled)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}, ensure_ascii=False) + "\n")

    print(f"[XGB] 保留 Top {K} 特征后，结果已保存到: {OUTPUT_FILE}")
    return model, topk_idx

if __name__ == "__main__":
    train_and_predict_xgb()
