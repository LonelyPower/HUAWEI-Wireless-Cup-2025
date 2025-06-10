import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import make_scorer, mean_squared_error

from data_and_preprocess import (
    load_train_data,
    load_test_data,
    compute_sample_weights,
    Preprocessor
)

# === 配置 ===
TOP_K = 650
TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE  = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE = "results_gridsearch_lgb.jsonl"


def rmse_scorer():
    return make_scorer(lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), greater_is_better=False)


def run_gridsearch(X, y, w):
    """在给定数据上执行超参数搜索，返回最佳模型"""
    param_grid = {
        'learning_rate': [0.005, 0.01],
        'num_leaves': [64, 128, 256],
        'min_child_samples': [10, 20, 50],
        'feature_fraction': [0.8, 1.0],
    }

    model = LGBMRegressor(
        objective='regression',
        metric='rmse',
        bagging_fraction=0.8,
        bagging_freq=5,
        n_estimators=2000,
        random_state=42,
        verbose=-1
    )

    grid = GridSearchCV(
        model,
        param_grid,
        scoring=rmse_scorer(),
        cv=3,
        n_jobs=-1,
        verbose=2
    )
    grid.fit(X, y, sample_weight=w)
    print(f"\nBest Params: {grid.best_params_}")
    print(f"Best CV RMSE: {-grid.best_score_:.5f}")
    return grid.best_estimator_


def main():
    # === 加载训练数据 ===
    X_avg, y_avg, _, y_raw = load_train_data(TRAIN_FILE)
    X_test_raw = load_test_data(TEST_FILE)
    w = compute_sample_weights(y_raw)

    # === 预处理 ===
    pre = Preprocessor(sparse_thresh=0.05, var_thresh=1e-6)
    X_scaled = pre.fit_transform(X_avg)
    X_test_scaled = pre.transform(X_test_raw)

    # === ExtraTrees 选 TOP_K 特征 ===
    et = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    et.fit(X_scaled, y_avg)
    importances = et.feature_importances_
    topk_idx = np.argsort(importances)[::-1][:TOP_K]
    X_sel = X_scaled[:, topk_idx]
    X_test_sel = X_test_scaled[:, topk_idx]

    # === 训练集划分 + 权重归一化 ===
    X_tr, X_val, y_tr, y_val, w_tr, w_val = train_test_split(
        X_sel, y_avg, w, test_size=0.1, random_state=42
    )
    w_tr = w_tr / np.mean(w_tr)
    w_val = w_val / np.mean(w_val)

    # === 网格搜索最优参数 ===
    best_model = run_gridsearch(X_tr, y_tr, w_tr)

    # === 验证集评估 ===
    y_val_pred = best_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"验证集 RMSE: {val_rmse:.5f}")

    # === 测试集预测并保存 ===
    preds = best_model.predict(X_test_sel)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for p in preds:
            f.write(json.dumps({'PredictResult': float(p)}, ensure_ascii=False) + '\n')
    print(f"结果写入: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
