import numpy as np  # 必须添加这一行，否则 np.where 会报错
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 1. 生成 100 个样本，2 个特征，分为 2 类
X, y = make_blobs(n_samples=100, centers=2, n_features=2, cluster_std=1.0, random_state=42)

# 2. 将标签从 [0, 1] 转换为感知机需要的 [-1, 1]
y = np.where(y == 0, -1, 1)

# 3. 打印前 5 行看看数据长什么样
print("特征 (X) 前5行:\n", X[:5])
print("标签 (y) 前5行:\n", y[:5])

# 4. 可视化（选做，能帮你直观看到数据）
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
plt.title("Generated Linear Data")
plt.show()