# data_and_preprocess.py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import VarianceThreshold

# 自动定位到项目根目录下的 datasets/AI-dataset
# BASE_DIR = Path(__file__).resolve().parent.parent

def load_train_data(path):
    """
    返回：
      X_avg: (n_samples, n_features)
      y_avg: (n_samples,)
      X_raw: (n_samples, 3, n_features)
      y_raw: (n_samples, 3)
    """
    with open(path, 'r', encoding='utf-8') as f:
        df = pd.read_json(f, lines=True)
    # 把 list 转成 np.ndarray 并 stack
    X_raw = np.stack(df['Features'].map(np.array).to_list())  # (n,3,D)
    y_raw = np.stack(df['Labels'].map(np.array).to_list())    # (n,3)
    X_avg = X_raw.mean(axis=1)                                 # (n,D)
    y_avg = y_raw.mean(axis=1)                                 # (n,)
    return X_avg, y_avg, X_raw, y_raw

def load_test_data(path):
    """返回 X_test: (n_samples, n_features)"""
    with open(path, 'r', encoding='utf-8') as f:
        df = pd.read_json(f, lines=True)
    X_test = np.stack(df['Feature'].map(np.array).to_list())
    return X_test

def compute_sample_weights(y_raw: np.ndarray,
                           std_thresh: float = 1.0,
                           delta_thresh: float = 2e7) -> np.ndarray:
    """
    根据三次采样标签的标准差和极差给样本加权：
      - std 和 delta 都高 → weight=0.3
      - std 或 delta 高 → weight=0.6
      - 否则 → weight=1.0
    """
    std   = y_raw.std(axis=1)
    delta = y_raw.max(axis=1) - y_raw.min(axis=1)
    w = np.where((std >= std_thresh) & (delta >= delta_thresh), 0.3,
         np.where((std >= std_thresh) | (delta >= delta_thresh), 0.6, 1.0))
    return w.astype(np.float32)

class Preprocessor:
    """
    - 稀疏特征剔除（zero_ratio > 1 - sparse_thresh）
    - 低方差特征剔除（VarianceThreshold）
    - log1p 变换 + RobustScaler
    """
    def __init__(self, sparse_thresh: float = 0.05, var_thresh: float = 1e-6):
        self.sparse_thresh = sparse_thresh
        self.var_thresh    = var_thresh
        self.scaler        = RobustScaler()
        self.mask_sparse   = None
        self.mask_var      = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        # 1. 稀疏筛选
        nonzero_ratio = (X != 0).mean(axis=0)
        mask1 = nonzero_ratio > self.sparse_thresh
        X1 = X[:, mask1]

        # 2. 低方差剔除
        vt = VarianceThreshold(threshold=self.var_thresh)
        X2 = vt.fit_transform(X1)

        # 3. log + RobustScaler
        X_log    = np.log1p(X2)
        X_scaled = self.scaler.fit_transform(X_log)

        # 保存 mask
        self.mask_sparse = mask1
        self.mask_var    = vt.get_support()
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        X1     = X[:, self.mask_sparse]
        X2     = X1[:, self.mask_var]
        X_log  = np.log1p(X2)
        return self.scaler.transform(X_log)
