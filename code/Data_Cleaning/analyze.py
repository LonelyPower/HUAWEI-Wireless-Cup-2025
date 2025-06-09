file_path = "../datasets/AI-dataset/data_train.jsonl"
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

# ========= ✅ 设置支持中文字体 =========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False


# ========= 📊 数据加载 =========
features_all = []
labels_all = []
label_stds = []
feature_deltas = []

with open(file_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        ftrs = np.array(item['Features'], dtype=np.float32)  # shape: (3, 1209)
        lbls = np.array(item['Labels'], dtype=np.float32)    # shape: (3,)
        
        features_all.append(ftrs)
        labels_all.append(lbls)
        label_stds.append(np.std(lbls))

        deltas = np.max(ftrs, axis=0) - np.min(ftrs, axis=0)
        feature_deltas.append(np.mean(deltas))

# 拼接整合
features_all = np.concatenate(features_all, axis=0)  # shape: (8334, 1209)
labels_all = np.concatenate(labels_all, axis=0)      # shape: (8334,)
label_stds = np.array(label_stds)
feature_deltas = np.array(feature_deltas)

# ========= 📄 统计汇总 =========
stats_df = pd.DataFrame({
    "标签标准差": label_stds,
    "特征平均波动": feature_deltas
})
stats_df["样本编号"] = range(1, len(stats_df)+1)
stats_df = stats_df[["样本编号", "标签标准差", "特征平均波动"]]

# ========= 📊 可视化 =========

plt.figure(figsize=(10, 4))
sns.histplot(label_stds, bins=40, kde=True)
plt.title("每样本标签三次采样的标准差分布")
plt.xlabel("标签标准差")
plt.ylabel("样本数量")
plt.grid(True)
plt.tight_layout()
plt.savefig("label_std_distribution.png")
plt.show()

plt.figure(figsize=(10, 4))
sns.histplot(feature_deltas, bins=40, kde=True, color='green')
plt.title("每样本特征三次采样的平均波动分布")
plt.xlabel("特征波动均值")
plt.ylabel("样本数量")
plt.grid(True)
plt.tight_layout()
plt.savefig("feature_delta_distribution.png")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(x=feature_deltas, y=label_stds)
plt.xlabel("特征波动均值")
plt.ylabel("标签标准差")
plt.title("特征波动 vs 标签不稳定性")
plt.grid(True)
plt.tight_layout()
plt.savefig("scatter_feature_vs_labelstd.png")
plt.show()

# ========= 💾 输出结果 =========
stats_df.to_csv("样本清洗统计.csv", index=False, encoding="utf-8-sig")

print("✅ 分析完成：")
print("- 生成图像文件: label_std_distribution.png, feature_delta_distribution.png, scatter_feature_vs_labelstd.png")
print("- 输出统计数据: 样本清洗统计.csv")
