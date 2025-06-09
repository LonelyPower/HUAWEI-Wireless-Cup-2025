import pandas as pd

# 加载分析结果 CSV（请确认文件路径）
df = pd.read_csv("样本清洗统计.csv")

# 提取各类别样本
stable = df[(df["标签标准差"] < 1.0) & (df["特征平均波动"] < 2e7)]
label_good_feat_bad = df[(df["标签标准差"] < 1.0) & (df["特征平均波动"] >= 2e7)]
label_bad_feat_good = df[(df["标签标准差"] >= 1.0) & (df["特征平均波动"] < 2e7)]
both_bad = df[(df["标签标准差"] >= 1.0) & (df["特征平均波动"] >= 2e7)]

# 输出结果保存
stable.to_csv("稳定样本.csv", index=False, encoding='utf-8-sig')
label_good_feat_bad.to_csv("特征异常_标签正常.csv", index=False, encoding='utf-8-sig')
label_bad_feat_good.to_csv("标签异常_特征正常.csv", index=False, encoding='utf-8-sig')
both_bad.to_csv("双重异常样本.csv", index=False, encoding='utf-8-sig')

print("✅ 筛选完成：生成4个分类文件：")
print("- 稳定样本.csv")
print("- 特征异常_标签正常.csv")
print("- 标签异常_特征正常.csv")
print("- 双重异常样本.csv")
