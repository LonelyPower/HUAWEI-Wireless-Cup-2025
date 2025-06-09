import numpy as np
import copy

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(np.float32)

def init_layers(layer_sizes, seed=42):
    np.random.seed(seed)
    params = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])
        b = np.zeros((1, layer_sizes[i+1]), dtype=np.float32)
        params.append((W, b))
    return params

def forward(X, params):
    A = X
    caches = []
    for idx, (W, b) in enumerate(params):
        Z = A @ W + b
        caches.append((A, Z, W, b))
        A = relu(Z) if idx < len(params)-1 else Z
    return A, caches

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

def get_batches(X, y, weights, batch_size, shuffle=True):
    m = X.shape[0]
    idxs = np.arange(m)
    if shuffle:
        np.random.shuffle(idxs)
    for start in range(0, m, batch_size):
        end = start + batch_size
        bi = idxs[start:end]
        yield X[bi], y[bi], weights[bi]
