import numpy as np
import json

def load_train_data(path):
    X_avg_list, y_avg_list = [], []
    X_raw_list, y_raw_list = [], []
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                ftrs = np.array(item['Features'], dtype=np.float32)  # (3, D)
                lbls = np.array(item['Labels'], dtype=np.float32)    # (3,)
                X_raw_list.append(ftrs)
                y_raw_list.append(lbls)
                X_avg_list.append(np.mean(ftrs, axis=0))  # (D,)
                y_avg_list.append(np.mean(lbls))
            except (KeyError, json.JSONDecodeError) as e:
                print(f"数据解析错误: {e}，跳过该行")
    return (
        np.stack(X_avg_list), np.array(y_avg_list),
        np.stack(X_raw_list), np.stack(y_raw_list)
    )

def load_test_data(path):
    feats = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            feats.append(np.array(item['Feature'], dtype=np.float32))
    return np.stack(feats)

def compute_weights_from_raw(X_raw, y_raw, std_thresh=1.0, delta_thresh=2e7):
    label_stds = np.std(y_raw, axis=1)
    feature_deltas = np.mean(np.max(X_raw, axis=1) - np.min(X_raw, axis=1), axis=1)
    w = np.ones_like(label_stds, dtype=np.float32)
    for i in range(len(w)):
        if label_stds[i] >= std_thresh and feature_deltas[i] >= delta_thresh:
            w[i] = 0.3
        elif label_stds[i] >= std_thresh or feature_deltas[i] >= delta_thresh:
            w[i] = 0.6
    return w

def compute_sparse_mask(X, thresh=0.05):
    nonzero_ratio = (X != 0).mean(axis=0)
    return nonzero_ratio > thresh

def robust_scaler_fit(X):
    median = np.median(X, axis=0, keepdims=True)
    q1 = np.percentile(X, 25, axis=0, keepdims=True)
    q3 = np.percentile(X, 75, axis=0, keepdims=True)
    iqr = q3 - q1
    iqr[iqr < 1e-2] = 1.0
    return median, iqr

def robust_scaler_transform(X, median, iqr):
    return (X - median) / iqr
