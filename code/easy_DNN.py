import numpy as np
import json

TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE = "results.jsonl"

# 1. 读取训练数据（包含3次采样，并取平均）
def load_train_data(path):
    features_all, labels_all = [], []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            ftrs = np.array(item['Features'], dtype=np.float32)  # shape: (3, n_features)
            lbls = np.array(item['Labels'], dtype=np.float32)    # shape: (3,)
            features_all.append(np.mean(ftrs, axis=0))
            labels_all.append(np.mean(lbls))
    return np.stack(features_all), np.array(labels_all)

# 2. 读取测试数据（单次采样）
def load_test_data(path):
    features = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            features.append(np.array(item['Feature'], dtype=np.float32))
    return np.stack(features)

# 3. ReLU 激活及其导数
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

# 4. 初始化模型参数
def init_layers(layer_sizes, seed=42):
    np.random.seed(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]), dtype=np.float32)
        params.append((W, b))
    return params

# 5. 前向传播
def forward(X, params):
    A = X
    caches = []
    for idx, (W, b) in enumerate(params):
        Z = A @ W + b
        caches.append((A, Z, W, b))
        A = relu(Z) if idx < len(params) - 1 else Z
    return A, caches

# 6. 反向传播
def backward(y_pred, y_true, caches):
    m = y_pred.shape[0]
    y_true = y_true.reshape(y_pred.shape)  # 确保形状一致 (n,1)
    dZ = (y_pred - y_true) / m

    grads = []
    for i in reversed(range(len(caches))):
        A_prev, Z, W, b = caches[i]
        dW = A_prev.T @ dZ
        db = np.sum(dZ, axis=0, keepdims=True)
        grads.insert(0, (dW, db))
        if i > 0:
            dA_prev = dZ @ W.T
            Z_prev = caches[i - 1][1]
            dZ = dA_prev * relu_derivative(Z_prev)
    return grads

# 7. 参数更新
def update(params, grads, lr):
    new_params = []
    for (W, b), (dW, db) in zip(params, grads):
        W = W - lr * dW
        b = b - lr * db
        new_params.append((W, b))
    return new_params

# 8. 特征标准化
def normalize_features(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

# 9. 模型训练主流程（包含归一化、梯度裁剪、nan/inf 检测）
def train(X, y, layers, epochs=200, lr=1e-4, clip_value=5.0):
    # 标准化
    X_norm, X_mean, X_std = normalize_features(X)
    params = init_layers(layers)
    y = y.reshape(-1, 1)

    for epoch in range(1, epochs + 1):
        # 前向
        y_pred, caches = forward(X_norm, params)
        # 计算 loss
        loss = np.mean((y_pred - y) ** 2)
        if np.isnan(loss) or np.isinf(loss):
            print(f"[Epoch {epoch}] 停止训练：检测到非法 loss={loss}")
            break
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] loss = {loss:.6e}")
        # 反向 + 梯度裁剪
        grads = backward(y_pred, y, caches)
        grads = [
            (np.clip(dW, -clip_value, clip_value),
             np.clip(db, -clip_value, clip_value))
            for dW, db in grads
        ]
        # 更新
        params = update(params, grads, lr)

    return params, X_mean, X_std

# 10. 模型预测
def predict(X, params, X_mean, X_std):
    X_norm = (X - X_mean) / X_std
    y_pred, _ = forward(X_norm, params)
    return y_pred.squeeze()

# 11. 保存结果为 jsonl

def save_results(preds, path):
    with open(path, 'w') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}) + '\n')


# ===== 主执行 =====
if __name__ == "__main__":
    X_train, y_train = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)
    print(f"Train X: {X_train.shape}, y: {y_train.shape}")

    layers = [X_train.shape[1], 256, 128, 1]
    params, X_mean, X_std = train(
        X_train, y_train, layers,
        epochs=200, lr=1e-4, clip_value=5.0
    )

    preds = predict(X_test, params, X_mean, X_std)
    save_results(preds, OUTPUT_FILE)
