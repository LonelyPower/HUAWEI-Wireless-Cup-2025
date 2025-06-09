import numpy as np
import json

TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE = "results1.jsonl"

# ========== 数据加载（保留原始三采样） ==========
def load_train_data(path):
    X_avg_list = []
    y_avg_list = []
    X_raw_list = []
    y_raw_list = []

    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                ftrs = np.array(item['Features'], dtype=np.float32)
                lbls = np.array(item['Labels'], dtype=np.float32)
                X_raw_list.append(ftrs)
                y_raw_list.append(lbls)
                X_avg_list.append(np.mean(ftrs, axis=0))
                y_avg_list.append(np.mean(lbls))
            except (KeyError, json.JSONDecodeError) as e:
                print(f"数据解析错误: {e}，跳过该行")

    return (
        np.stack(X_avg_list),
        np.array(y_avg_list),
        np.stack(X_raw_list),
        np.stack(y_raw_list)
    )

# ========== 读取测试集 ==========
def load_test_data(path):
    features = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            features.append(np.array(item['Feature'], dtype=np.float32))
    return np.stack(features)

# ========== ReLU 及导数 ==========
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

# ========== 初始化网络参数 ==========
def init_layers(layer_sizes, seed=42):
    np.random.seed(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]))
        params.append((W, b))
    return params

# ========== 前向传播 ==========
def forward(X, params):
    A = X
    caches = []
    for idx, (W, b) in enumerate(params):
        Z = A @ W + b
        caches.append((A, Z, W, b))
        A = relu(Z) if idx < len(params) - 1 else Z
    return A, caches

# ========== 反向传播 ==========
def backward(y_pred, y_true, caches, weights, clip_value=5.0):
    m = y_true.shape[0]
    y_true = y_true.reshape(y_pred.shape)
    dZ = (y_pred - y_true) * weights.reshape(-1, 1) / m
    grads = []

    for i in reversed(range(len(caches))):
        A_prev, Z, W, b = caches[i]
        dW = np.clip(A_prev.T @ dZ, -clip_value, clip_value)
        db = np.clip(np.sum(dZ, axis=0, keepdims=True), -clip_value, clip_value)
        grads.insert(0, (dW, db))
        if i > 0:
            dA_prev = dZ @ W.T
            _, Z_prev, _, _ = caches[i - 1]
            dZ = dA_prev * relu_derivative(Z_prev)
    return grads

# ========== 参数更新 ==========
def update(params, grads, lr):
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]

# ========== 特征标准化 + 维度选择 ==========
def normalize_features(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    too_small = std < 1e-2
    too_large = std > 1e6
    mask = ~(too_small | too_large)
    X_sel = X[:, mask[0]]
    mean_sel = mean[:, mask[0]]
    std_sel = std[:, mask[0]]
    std_sel[std_sel < 1e-2] = 1.0
    return (X_sel - mean_sel) / std_sel, mean_sel, std_sel, mask

# ========== 置信度权重计算 ==========
def compute_weights_from_raw(X_raw, y_raw, std_thresh=1.0, delta_thresh=2e7):
    label_stds = np.std(y_raw, axis=1)
    feature_deltas = np.mean(np.max(X_raw, axis=1) - np.min(X_raw, axis=1), axis=1)
    weights = np.ones_like(label_stds, dtype=np.float32)
    for i in range(len(weights)):
        if label_stds[i] >= std_thresh and feature_deltas[i] >= delta_thresh:
            weights[i] = 0.3
        elif label_stds[i] >= std_thresh or feature_deltas[i] >= delta_thresh:
            weights[i] = 0.6
        else:
            weights[i] = 1.0
    return weights

# ========== 模型训练 ==========
def train(X, y, X_raw, y_raw, layers, epochs=200, lr=1e-4, clip_value=5.0):
    X_norm, mean, std, mask = normalize_features(X)
    params = init_layers([X_norm.shape[1]] + layers[1:])
    y = y.reshape(-1, 1)
    weights = compute_weights_from_raw(X_raw, y_raw)

    for epoch in range(1, epochs + 1):
        y_pred, caches = forward(X_norm, params)
        loss = 0.5 * np.mean(weights.reshape(-1, 1) * ((y_pred - y) ** 2))
        grads = backward(y_pred, y, caches, weights, clip_value)
        params = update(params, grads, lr)
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] loss = {loss:.6e}, y_pred std = {y_pred.std():.6f}")
    return params, mean, std, mask

# ========== 模型预测 ==========
def predict(X, params, mean, std, mask):
    X_sel = X[:, mask[0]]
    X_norm = (X_sel - mean) / std
    y_pred, _ = forward(X_norm, params)
    return y_pred.squeeze()

# ========== 保存预测结果 ==========
def save_results(preds, path):
    with open(path, 'w') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}) + '\n')

# ========== 主程序入口 ==========
if __name__ == "__main__":
    X_avg, y_avg, X_raw, y_raw = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)
    print(f"训练数据: X.shape={X_avg.shape}, y.shape={y_avg.shape}")

    layers = [X_avg.shape[1], 256, 128, 1]
    params, mean, std, mask = train(X_avg, y_avg, X_raw, y_raw, layers)
    preds = predict(X_test, params, mean, std, mask)
    save_results(preds, OUTPUT_FILE)
    print(f"预测结果已保存至 {OUTPUT_FILE}")
