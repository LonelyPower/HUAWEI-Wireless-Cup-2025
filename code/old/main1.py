# main.py
import json
import copy
import numpy as np
from code.data_and_preprocess1 import *
from code.model1 import *

TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE = "results1.jsonl"

def train(X_avg, y_avg, X_raw, y_raw,
          layers, epochs=200, initial_lr=1e-4, batch_size=64,
          val_ratio=0.1, beta1=0.9, beta2=0.999, eps=1e-8,
          lr_decay=0.5, decay_step=50, patience=10, clip_value=5.0,
          seed=42, top_k_features=300):
    # 1) 稀疏度筛选
    sparse_mask = compute_sparse_mask(X_avg, thresh=0.05)
    X_sel = X_avg[:, sparse_mask]
    X_raw_sel = X_raw[:, :, sparse_mask]

    # 2) 对数变换 + RobustScaler（全体被稀疏筛选后的特征）
    X_log = np.log1p(X_sel)
    median_sel, iqr_sel = robust_scaler_fit(X_log)
    X_norm = robust_scaler_transform(X_log, median_sel, iqr_sel)
    y = y_avg.reshape(-1)

    # 3) 基于决策桩的特征重要性打分
    importances = compute_feature_importances_decision_stump(X_norm, y, n_thresholds=10)
    idx_sorted = np.argsort(-importances)
    idx_topk = idx_sorted[:top_k_features]

    # 构造最终特征掩码
    sparse_idx = np.where(sparse_mask)[0]
    selected_idx = sparse_idx[idx_topk]
    final_mask = np.zeros_like(sparse_mask)
    final_mask[selected_idx] = True

    # 用 final_mask 重建数据并重新计算 scaler
    X_masked = X_avg[:, final_mask]
    X_raw_masked = X_raw[:, :, final_mask]
    X_log_m = np.log1p(X_masked)
    # **关键修改：针对最终选出的特征重新 fit scaler**
    median_m, iqr_m = robust_scaler_fit(X_log_m)
    X_norm_m = robust_scaler_transform(X_log_m, median_m, iqr_m)

    weights_all = compute_weights_from_raw(X_raw_masked, y_raw)

    # 4) 划分训练/验证
    np.random.seed(seed)
    m = X_norm_m.shape[0]
    idxs = np.arange(m)
    np.random.shuffle(idxs)
    val_size = int(m * val_ratio)
    val_idx, train_idx = idxs[:val_size], idxs[val_size:]
    X_val, y_val, w_val = X_norm_m[val_idx], y[val_idx], weights_all[val_idx]
    X_tr,  y_tr,  w_tr  = X_norm_m[train_idx], y[train_idx], weights_all[train_idx]

    # 5) 初始化模型参数
    params = init_layers([X_norm_m.shape[1]] + layers[1:], seed)
    mW = [np.zeros_like(W) for W,_ in params]
    vW = [np.zeros_like(W) for W,_ in params]
    mB = [np.zeros_like(b) for _,b in params]
    vB = [np.zeros_like(b) for _,b in params]
    best_params = copy.deepcopy(params)
    best_val = float('inf')
    wait = 0
    lr = initial_lr
    iters = 0

    # 6) 训练循环
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        tot = 0
        for Xb, yb, wb in get_batches(X_tr, y_tr, w_tr, batch_size):
            iters += 1
            y_pred, caches = forward(Xb, params)
            train_loss += 0.5 * np.mean(wb.reshape(-1,1)*(y_pred-yb.reshape(-1,1))**2) * Xb.shape[0]
            tot += Xb.shape[0]
            grads = backward(y_pred, yb.reshape(-1,1), caches, wb, clip_value)
            # Adam 更新
            for i, ((W,b),(dW,db)) in enumerate(zip(params, grads)):
                mW[i] = beta1*mW[i] + (1-beta1)*dW
                vW[i] = beta2*vW[i] + (1-beta2)*(dW**2)
                mB[i] = beta1*mB[i] + (1-beta1)*db
                vB[i] = beta2*vB[i] + (1-beta2)*(db**2)
                mW_corr = mW[i]/(1-beta1**iters)
                vW_corr = vW[i]/(1-beta2**iters)
                mB_corr = mB[i]/(1-beta1**iters)
                vB_corr = vB[i]/(1-beta2**iters)
                W -= lr * mW_corr/(np.sqrt(vW_corr)+eps)
                b -= lr * mB_corr/(np.sqrt(vB_corr)+eps)
                params[i] = (W, b)
        train_loss /= tot

        # 验证
        yv_pred, _ = forward(X_val, params)
        val_loss = 0.5 * np.mean(w_val.reshape(-1,1)*(yv_pred - y_val.reshape(-1,1))**2)
        print(f"[{epoch}] train={train_loss:.6e}  val={val_loss:.6e}  lr={lr:.2e}")

        # 早停 & 学习率衰减
        if val_loss < best_val:
            best_val, best_params = val_loss, copy.deepcopy(params)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}, best_val={best_val:.6e}")
                break
        if epoch % decay_step == 0:
            lr *= lr_decay
            print(f" lr decayed to {lr:.2e}")

    # 返回：最优参数 + 针对最终特征的 scaler + 特征掩码
    return best_params, median_m, iqr_m, final_mask

def predict(X, params, median, iqr, feature_mask):
    X_sel = X[:, feature_mask]
    X_log = np.log1p(X_sel)
    X_norm = robust_scaler_transform(X_log, median, iqr)
    y_pred, _ = forward(X_norm, params)
    return y_pred.squeeze()

def save_results(preds, path):
    with open(path, 'w') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}) + "\n")

if __name__ == "__main__":
    X_avg, y_avg, X_raw, y_raw = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)
    print(f"训练集: X_avg.shape={X_avg.shape}, y_avg.shape={y_avg.shape}")
    layers = [X_avg.shape[1], 256, 128, 1]
    params, median, iqr, feat_mask = train(
        X_avg, y_avg, X_raw, y_raw, layers,
        epochs=200, initial_lr=1e-4, top_k_features=300
    )
    preds = predict(X_test, params, median, iqr, feat_mask)
    save_results(preds, OUTPUT_FILE)
    print(f"预测结果已保存至 {OUTPUT_FILE}")
