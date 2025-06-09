import numpy as np
import json

TRAIN_FILE = "../datasets/AI-dataset/data_train.jsonl"
TEST_FILE = "../datasets/AI-dataset/data_test.jsonl"
OUTPUT_FILE = "results1.jsonl"

# 1. 读取训练数据
def load_train_data(path):
    features_all, labels_all = [], []
    with open(path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                ftrs = np.array(item['Features'], dtype=np.float32)
                lbls = np.array(item['Labels'], dtype=np.float32)
                features_all.append(np.mean(ftrs, axis=0))
                labels_all.append(np.mean(lbls))
            except (KeyError, json.JSONDecodeError) as e:
                print(f"数据解析错误: {e}，跳过该行")
    return np.stack(features_all), np.array(labels_all)

# 2. 读取测试数据
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

# 4. 初始化模型参数（修正语法错误）
def init_layers(layer_sizes, seed=42):
    assert all(s > 0 for s in layer_sizes), "层大小必须为正整数"
    np.random.seed(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        # 修正初始化方式
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]))  # 修正语法
        params.append((W, b))  # 修正：添加括号
    return params

# 5. 前向传播（修正缓存存储方式）
def forward(X, params):
    A = X
    caches = []
    for idx, (W, b) in enumerate(params):
        Z = A @ W + b
        if np.isnan(Z).any():
            raise ValueError(f"第{idx}层输出出现NaN")
        # 修正：正确存储缓存为元组
        caches.append((A, Z, W, b))  # 修正：添加括号
        A = relu(Z) if idx < len(params) - 1 else Z
    return A, caches

# 6. 反向传播（关键修正：使用前一层Z）
def backward(y_pred, y_true, caches, clip_value=5.0):
    m = y_pred.shape[0]
    y_true = y_true.reshape(y_pred.shape)
    dZ = (y_pred - y_true) / m  # 输出层梯度

    grads = []
    # 按层逆序处理：从输出层开始
    for i in reversed(range(len(caches))):
        A_prev, Z, W, b = caches[i]
        dW = np.clip(A_prev.T @ dZ, -clip_value, clip_value)
        db = np.clip(np.sum(dZ, axis=0, keepdims=True), -clip_value, clip_value)
        grads.insert(0, (dW, db))
        
        if i > 0:  # 计算前一层的梯度
            dA_prev = dZ @ W.T
            # 关键修正：获取前一层（i-1）的Z
            _, Z_prev, _, _ = caches[i-1]  # 取出前一层激活前的Z
            dZ = dA_prev * relu_derivative(Z_prev)  # 使用前一层Z计算导数
    return grads

# 7. 参数更新
def update(params, grads, lr):
    return [(W - lr * dW, b - lr * db) for (W, b), (dW, db) in zip(params, grads)]

# 8. 特征标准化
def normalize_features(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std

# 9. 模型训练
def train(X, y, layers, epochs=200, lr=1e-4, clip_value=5.0):
    X_norm, X_mean, X_std = normalize_features(X)
    params = init_layers(layers)
    y = y.reshape(-1, 1)

    for epoch in range(1, epochs + 1):
        y_pred, caches = forward(X_norm, params)
        loss = 0.5 * np.mean((y_pred - y) ** 2)
        
        if np.isnan(loss):
            print(f"[Epoch {epoch}] 检测到NaN，停止训练")
            break
            
        grads = backward(y_pred, y, caches, clip_value)
        params = update(params, grads, lr)
        
        if epoch % 10 == 0:
            print(f"[Epoch {epoch}] loss = {loss:.6e}")

    return params, X_mean, X_std

# 10. 模型预测
def predict(X, params, X_mean, X_std):
    X_norm = (X - X_mean) / X_std
    y_pred, _ = forward(X_norm, params)
    return y_pred.squeeze()

# 11. 保存结果
def save_results(preds, path):
    with open(path, 'w') as f:
        for p in preds:
            f.write(json.dumps({"PredictResult": float(p)}) + '\n')

if __name__ == "__main__":
    # 数据加载
    X_train, y_train = load_train_data(TRAIN_FILE)
    X_test = load_test_data(TEST_FILE)
    print(f"训练数据: X.shape={X_train.shape}, y.shape={y_train.shape}")
    print(f"测试数据: X.shape={X_test.shape}")

    # 模型配置
    layers = [X_train.shape[1], 256, 128, 1]
    
    # 训练
    params, X_mean, X_std = train(
        X_train, y_train, layers,
        epochs=200,
        lr=1e-4,
        clip_value=5.0
    )

    # 预测与保存
    preds = predict(X_test, params, X_mean, X_std)
    save_results(preds, OUTPUT_FILE)
    print(f"预测结果已保存至 {OUTPUT_FILE}")