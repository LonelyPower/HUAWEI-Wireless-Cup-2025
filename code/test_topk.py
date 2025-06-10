
# -*- coding: utf-8 -*-
"""
test_topk.py

测试不同 Top-K 特征数对模型性能的影响，输出各 K 值对应的验证集 RMSE，
并选出最优 K，结果保存到 tune_results.jsonl。
"""

import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb

# 请确保 data_and_preprocess.py 中实现了以下接口：
# - load_train_data(train_file) -> X_avg, y_avg, X_raw, y_raw
# - load_test_data(test_file)  -> X_test_raw
# - compute_sample_weights(y_raw) -> w
# - Preprocessor 类：fit_transform(X) / transform(X)
from data_and_preprocess import (
    load_train_data,
    load_test_data,
    compute_sample_weights,
    Preprocessor
)


def train_with_k(X, y, w, X_test, K, random_state=42):
    """
    对给定的 K 值：
    1) 预处理原始特征；
    2) 用 ExtraTreesRegressor 选出前 K 个特征；
    3) 训练 LightGBM 并返回验证集 RMSE。
    """
    # 1. 预处理
    pre = Preprocessor(sparse_thresh=0.05, var_thresh=1e-6)
    X_scaled      = pre.fit_transform(X)
    X_test_scaled = pre.transform(X_test)

    # 2. 树模型选 K 个特征
    et = ExtraTreesRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    et.fit(X_scaled, y)
    importances = et.feature_importances_
    idx = np.argsort(importances)[::-1][:K]
    X_sel, X_test_sel = X_scaled[:, idx], X_test_scaled[:, idx]

    # 3. 划分训练/验证
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_sel, y, w, test_size=0.1, random_state=random_state
    )
    w_tr  = w_tr  / np.mean(w_tr)
    w_val = w_val / np.mean(w_val)

    # 4. 训练 LGBM
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
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    # 5. 返回验证集 RMSE
    val_rmse = model.best_score_['valid_0']['rmse']
    return val_rmse


def tune_top_k(train_file, test_file, candidate_K, random_state=42):
    """
    在 candidate_K 列表中的每个 K 上，都调用 train_with_k，
    打印并记录 (K, RMSE)，最后返回所有结果和最佳 K。
    """
    # 加载训练/测试数据
    X_avg, y_avg, _, y_raw = load_train_data(train_file)
    w = compute_sample_weights(y_raw)
    X_test = load_test_data(test_file)

    results = []
    for K in candidate_K:
        rmse = train_with_k(X_avg, y_avg, w, X_test, K, random_state)
        print(f"K = {K:4d} ➞  验证集 RMSE = {rmse:.4f}")
        results.append({'K': int(K), 'rmse': float(rmse)})

    # 找到最小 RMSE 对应的 K
    best = min(results, key=lambda x: x['rmse'])
    print(f"\n最佳 K = {best['K']}, 对应验证集 RMSE = {best['rmse']:.4f}")

    # 保存所有结果
    out_path = Path("tune_results.jsonl")
    with out_path.open('w', encoding='utf-8') as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"所有试验结果已保存到 {out_path.resolve()}")

    return results, best['K']


if __name__ == "__main__":
    # 候选的 K 值列表，可根据需要增删
    candidate_K = [550,600,650,700]
    TRAIN_FILE  = "../datasets/AI-dataset/data_train.jsonl"
    TEST_FILE   = "../datasets/AI-dataset/data_test.jsonl"

    tune_top_k(
        train_file=TRAIN_FILE,
        test_file=TEST_FILE,
        candidate_K=candidate_K,
        random_state=42
    )
