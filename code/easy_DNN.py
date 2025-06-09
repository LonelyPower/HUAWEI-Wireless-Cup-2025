import numpy as np
import json
import copy

TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE = "results1.jsonl"
def pca_fit(X, variance_ratio=0.95):
    """
    在 X（N×D）的基础上做 PCA，找出能覆盖 variance_ratio 方差的前 k 个主成分
    返回 X 的均值和 principal components（k×D）
    """
    # 1. 均值中心化
    mean = X.mean(axis=0, keepdims=True)               # (1, D)
    Xc = X - mean                                      # (N, D)
    # 2. SVD 分解
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt: (D, D)
    # 3. 方差贡献率
    var = S**2
    cumvar = np.cumsum(var) / np.sum(var)
    k = np.searchsorted(cumvar, variance_ratio) + 1
    # 4. 取前 k 个主成分
    components = Vt[:k]                                # (k, D)
    return mean, components

def pca_transform(X, mean, components):
    """
    对新数据 X 做同样的中心化 + 投影
    返回降维后 X_pca (N, k)
    """
    Xc = X - mean                                       # (N, D)
    return Xc @ components.T                           # (N, k)


# ========== 数据加载（保留原始三采样） ==========
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
        np.stack(X_avg_list),      # (N, D)
        np.array(y_avg_list),      # (N,)
        np.stack(X_raw_list),      # (N, 3, D)
        np.stack(y_raw_list)       # (N, 3)
    )

# ========== 读取测试集 ==========
def load_test_data(path):
    feats = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            feats.append(np.array(item['Feature'], dtype=np.float32))
    return np.stack(feats)  # (M, D)

# ========== 置信度权重计算 ==========
def compute_weights_from_raw(X_raw, y_raw, std_thresh=1.0, delta_thresh=2e7):
    label_stds = np.std(y_raw, axis=1)
    feature_deltas = np.mean(np.max(X_raw, axis=1) - np.min(X_raw, axis=1), axis=1)
    w = np.ones_like(label_stds, dtype=np.float32)
    for i in range(len(w)):
        if label_stds[i] >= std_thresh and feature_deltas[i] >= delta_thresh:
            w[i] = 0.3
        elif label_stds[i] >= std_thresh or feature_deltas[i] >= delta_thresh:
            w[i] = 0.6
    return w  # (N,)

# ========== 激活及其导数 ==========
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

# ========== 初始化网络参数 ==========
def init_layers(layer_sizes, seed=42):
    np.random.seed(seed)
    params = []
    for i in range(len(layer_sizes)-1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]), dtype=np.float32)
        params.append((W, b))
    return params

# ========== 前向传播 ==========
def forward(X, params):
    A = X
    caches = []
    for idx, (W, b) in enumerate(params):
        Z = A @ W + b
        caches.append((A, Z, W, b))
        A = relu(Z) if idx < len(params)-1 else Z
    return A, caches

# ========== 反向传播 ==========
def backward(y_pred, y_true, caches, weights=None, clip_value=5.0):
    m = y_true.shape[0]
    y_true = y_true.reshape(y_pred.shape)
    if weights is None:
        weights = np.ones((m,), dtype=np.float32)
    dZ = (y_pred - y_true) * weights.reshape(-1,1) / m
    grads = []
    for i in reversed(range(len(caches))):
        A_prev, Z, W, b = caches[i]
        dW = np.clip(A_prev.T @ dZ, -clip_value, clip_value)
        db = np.clip(np.sum(dZ, axis=0, keepdims=True), -clip_value, clip_value)
        grads.insert(0, (dW, db))
        if i > 0:
            dA_prev = dZ @ W.T
            _, Z_prev, _, _ = caches[i-1]
            dZ = dA_prev * relu_derivative(Z_prev)
    return grads

# ========== 生成 Mini-batches ==========
def get_batches(X, y, weights, batch_size, shuffle=True):
    m = X.shape[0]
    idxs = np.arange(m)
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, m, batch_size):
        end = start + batch_size
        bi = idxs[start:end]
        yield X[bi], y[bi], weights[bi]

# ========== 稀疏特征删除 ==========
def compute_sparse_mask(X, thresh=0.05):
    # 保留非零比例 > thresh 的列
    nonzero_ratio = (X != 0).mean(axis=0)
    return nonzero_ratio > thresh  # (D,)

# ========== RobustScaler ==========
def robust_scaler_fit(X):
    median = np.median(X, axis=0, keepdims=True)
    q1 = np.percentile(X, 25, axis=0, keepdims=True)
    q3 = np.percentile(X, 75, axis=0, keepdims=True)
    iqr = q3 - q1
    iqr[iqr < 1e-2] = 1.0
    return median, iqr

def robust_scaler_transform(X, median, iqr):
    return (X - median) / iqr

# ========== 训练主函数 ==========
def train(
    X_avg, y_avg, X_raw, y_raw, layers,
    epochs=200,
    initial_lr=1e-4,
    batch_size=64,
    val_ratio=0.1,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    lr_decay=0.5,
    decay_step=50,
    patience=10,
    clip_value=5.0,
    seed=42,
    variance_ratio=0.95
):
    # 1. 删除极度稀疏特征（非零比例 < 5%）
    nonzero_ratio = (X_avg != 0).mean(axis=0)
    sparse_mask = nonzero_ratio > 0.05            # (D,)
    X_sel = X_avg[:, sparse_mask]                  # (N, D')
    X_raw_sel = X_raw[:, :, sparse_mask]           # (N, 3, D')

    # 2. log1p 数值压缩
    X_log = np.log1p(X_sel)                        # (N, D')

    # 3. RobustScaler：中位数 + IQR
    median = np.median(X_log, axis=0, keepdims=True)     # (1, D')
    q1 = np.percentile(X_log, 25, axis=0, keepdims=True)
    q3 = np.percentile(X_log, 75, axis=0, keepdims=True)
    iqr = q3 - q1
    iqr[iqr < 1e-2] = 1.0
    X_norm = (X_log - median) / iqr              # (N, D')

    # 4. PCA 降维
    pca_mean, pca_components = pca_fit(X_norm, variance_ratio)
    X_pca = pca_transform(X_norm, pca_mean, pca_components)  # (N, k)
    X_input = X_pca

    # 5. 构造标签与权重
    y = y_avg.reshape(-1, 1)                     # (N, 1)
    # 这里假定 compute_weights_from_raw 已定义
    weights_all = compute_weights_from_raw(X_raw_sel, y_raw)  # (N,)

    # 6. 划分训练/验证集
    np.random.seed(seed)
    m = X_input.shape[0]
    idxs = np.arange(m)
    np.random.shuffle(idxs)
    val_size = int(m * val_ratio)
    val_idx, train_idx = idxs[:val_size], idxs[val_size:]
    X_val, y_val, w_val = X_input[val_idx], y[val_idx], weights_all[val_idx]
    X_tr,  y_tr,  w_tr  = X_input[train_idx], y[train_idx], weights_all[train_idx]

    # 7. 初始化网络 & Adam 状态
    params = init_layers([X_input.shape[1]] + layers[1:], seed)
    mW = [np.zeros_like(W) for W,_ in params]
    vW = [np.zeros_like(W) for W,_ in params]
    mB = [np.zeros_like(b) for _,b in params]
    vB = [np.zeros_like(b) for _,b in params]

    best_params, best_val = None, float('inf')
    wait, lr, iters = 0, initial_lr, 0

    # 8. 训练循环
    for epoch in range(1, epochs+1):
        train_loss, total = 0.0, 0
        for Xb, yb, wb in get_batches(X_tr, y_tr, w_tr, batch_size, shuffle=True):
            iters += 1
            # forward
            y_pred, caches = forward(Xb, params)
            train_loss += 0.5 * np.mean(wb.reshape(-1,1)*(y_pred-yb)**2) * Xb.shape[0]
            total += Xb.shape[0]
            # backward
            grads = backward(y_pred, yb, caches, wb, clip_value)
            # Adam update
            for i, ((W,b),(dW,db)) in enumerate(zip(params, grads)):
                mW[i] = beta1*mW[i] + (1-beta1)*dW
                vW[i] = beta2*vW[i] + (1-beta2)*(dW**2)
                mB[i] = beta1*mB[i] + (1-beta1)*db
                vB[i] = beta2*vB[i] + (1-beta2)*(db**2)

                mW_corr = mW[i] / (1 - beta1**iters)
                vW_corr = vW[i] / (1 - beta2**iters)
                mB_corr = mB[i] / (1 - beta1**iters)
                vB_corr = vB[i] / (1 - beta2**iters)

                W -= lr * mW_corr / (np.sqrt(vW_corr) + eps)
                b -= lr * mB_corr / (np.sqrt(vB_corr) + eps)
                params[i] = (W, b)

        train_loss /= total

        # 验证集
        yv_pred, _ = forward(X_val, params)
        val_loss = 0.5 * np.mean(w_val.reshape(-1,1)*(yv_pred-y_val)**2)

        print(f"[Epoch {epoch}] train={train_loss:.6e}  val={val_loss:.6e}  lr={lr:.2e}")

        # 早停 & lr 调度
        if val_loss < best_val:
            best_val, best_params, wait = val_loss, copy.deepcopy(params), 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}, best_val={best_val:.6e}")
                break

        if epoch % decay_step == 0:
            lr *= lr_decay
            print(f"  -> lr decayed to {lr:.2e}")

    return best_params, median, iqr, sparse_mask, pca_mean, pca_components
# ========== 预测函数 ==========
def predict(X, params, median, iqr, sparse_mask, pca_mean, pca_components):
    # 同样的预处理
    X_sel = X[:, sparse_mask]
    X_log = np.log1p(X_sel)
    X_norm = robust_scaler_transform(X_log, median, iqr)
    X_pca = pca_transform(X_norm, pca_mean, pca_components)
    # 3. 前向
    y_pred, _ = forward(X_pca, params)
    # y_pred, _ = forward(X_norm, params)
    return y_pred.squeeze()

# ========== 保存结果 ==========
def save_results(preds, path):
    with open(path, 'w') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}) + "\n")

# ========== 主程序 ==========
if __name__ == "__main__":
    X_avg, y_avg, X_raw, y_raw = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)
    print(f"训练集: X_avg.shape={X_avg.shape}, y_avg.shape={y_avg.shape}")

    layers = [X_avg.shape[1], 256, 128, 1]
    params, median, iqr, sparse_mask , pca_mean, pca_components= train(
        X_avg, y_avg, X_raw, y_raw, layers,
        epochs=200, initial_lr=1e-4, batch_size=64,
        val_ratio=0.1, lr_decay=0.5, decay_step=50, patience=10
    )

    preds = predict(X_test, params, median, iqr, sparse_mask, pca_mean, pca_components)
    save_results(preds, OUTPUT_FILE)
    print(f"预测结果已保存至 {OUTPUT_FILE}")
