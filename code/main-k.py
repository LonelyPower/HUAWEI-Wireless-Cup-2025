# main.py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
import lightgbm as lgb

from data_and_preprocess import (
    load_train_data, load_test_data,
    compute_sample_weights, Preprocessor
)

# 可调参数：Top K 特征数量
TOP_K = 650

OUTPUT_FILE = "results_lgb_topk.jsonl"
TRAIN_FILE  = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE   = "../datasets/AI-dataset/data_test.jsonl"

def train_and_predict(K=TOP_K, random_state=42):
    # 1. 加载数据
    X_avg, y_avg, X_raw, y_raw = load_train_data(TRAIN_FILE)  # :contentReference[oaicite:0]{index=0}
    X_test = load_test_data(TEST_FILE)                        # :contentReference[oaicite:1]{index=1}

    # 2. 样本权重
    w = compute_sample_weights(y_raw)                         # :contentReference[oaicite:2]{index=2}

    # 3. 特征预处理（稀疏 + 低方差 + log1p + RobustScaler）
    pre = Preprocessor(sparse_thresh=0.05, var_thresh=1e-6)   # :contentReference[oaicite:3]{index=3}
    X_scaled      = pre.fit_transform(X_avg)
    X_test_scaled = pre.transform(X_test)

    # 4. 树模型（ExtraTrees）特征筛选
    et = ExtraTreesRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    et.fit(X_scaled, y_avg)
    importances = et.feature_importances_
    topk_idx = np.argsort(importances)[::-1][:K]
    # 只保留 Top K 特征
    X_scaled      = X_scaled[:, topk_idx]
    X_test_scaled = X_test_scaled[:, topk_idx]

    # 5. 划分训练/验证，并归一化权重
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_scaled, y_avg, w, test_size=0.1, random_state=random_state
    )
    w_tr  = w_tr  / np.mean(w_tr)
    w_val = w_val / np.mean(w_val)

    # 6. 训练 LightGBM
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        learning_rate=0.01,
        num_leaves=128,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=20,
        min_split_gain=0.0,
        n_estimators=2000,
        random_state=random_state
    )
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )

    # 7. 对测试集做预测并写入文件
    preds = model.predict(X_test_scaled, num_iteration=model.best_iteration_)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}, ensure_ascii=False) + "\n")

    print(f"保留 Top {K} 特征后，预测结果已保存到: {OUTPUT_FILE}")
    return model, topk_idx

if __name__ == "__main__":
    train_and_predict()
