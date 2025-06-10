# main.py
import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from data_and_preprocess import (
    load_train_data, load_test_data,
    compute_sample_weights, Preprocessor
)

OUTPUT_FILE = "results_lgb.jsonl"
TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE = "../datasets/AI-dataset/data_test.jsonl"
def train_and_predict():
    # 1. 加载数据
    X_avg, y_avg, X_raw, y_raw = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)

    # 2. 样本权重
    w = compute_sample_weights(y_raw)

    # 3. 特征预处理
    pre = Preprocessor(sparse_thresh=0.05, var_thresh=1e-6)
    X_scaled      = pre.fit_transform(X_avg)
    X_test_scaled = pre.transform(X_test)

    # 4. 划分训练/验证，并归一化权重
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_scaled, y_avg, w, test_size=0.1, random_state=42
    )
    w_tr  = w_tr  / np.mean(w_tr)
    w_val = w_val / np.mean(w_val)

    # 5. 训练 LightGBM (sklearn 接口 + callbacks)
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
        random_state=42
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

    # 6. 预测并保存
    preds = model.predict(X_test_scaled, num_iteration=model.best_iteration_)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}, ensure_ascii=False) + "\n")
    print(f"预测结果已保存到: {OUTPUT_FILE}")

if __name__ == "__main__":
    train_and_predict()
