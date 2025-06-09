import json
import numpy as np

DATA_PATH = "../../datasets/AI-dataset/data_train.jsonl"  # 修改为你的训练集路径

def analyze_zero_feature_ratio(path):
    ratios = []
    with open(path, 'r') as f:
        for idx, line in enumerate(f, 1):
            try:
                item = json.loads(line)
                features = np.array(item["Features"], dtype=np.float32)  # shape: (3, D)
                # 统计三次采样合并后的特征（或取均值也可）
                avg_feat = np.mean(features, axis=0)
                zero_ratio = np.sum(avg_feat == 0) / avg_feat.size
                ratios.append(zero_ratio)
            except Exception as e:
                print(f"[Line {idx}] 错误: {e}")
    
    ratios = np.array(ratios)
    print("=== 零值特征比例统计 ===")
    print(f"样本总数：{len(ratios)}")
    print(f"平均零值比例：{ratios.mean():.4f}")
    print(f"最大零值比例：{ratios.max():.4f}")
    print(f"最小零值比例：{ratios.min():.4f}")
    print(f"零值比例 > 0.5 的样本数：{np.sum(ratios > 0.5)}")
    print(f"零值比例 > 0.9 的样本数：{np.sum(ratios > 0.9)}")

if __name__ == "__main__":
    analyze_zero_feature_ratio(DATA_PATH)
