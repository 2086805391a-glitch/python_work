import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 准备数据 (面积 vs 房价)
# 假设关系：房价 = 2 * 面积 + 0.5
x_train = np.array([[20], [40], [60], [80], [100], [120]], dtype=np.float32)
y_train = np.array([[45], [85], [125], [165], [205], [245]], dtype=np.float32)

# 转换为 PyTorch 的 Tensor 张量
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 定义简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # 输入维度为1 (面积)，输出维度为1 (价格)
        self.linear = nn.Linear(1, 1) 

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()


# 损失函数：均方误差 (MSE)
criterion = nn.MSELoss()

# 优化器：随机梯度下降 (SGD)，学习率为 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：模型预测
    outputs = model(x_train)
    loss = criterion(outputs, y_train)

    # 反向传播：计算梯度并优化
    optimizer.zero_grad() # 清空之前的梯度
    loss.backward()       # 计算现在的梯度
    optimizer.step()      # 更新权重

    if (epoch+1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')



# 切换到评估模式
model.eval()
predicted = model(x_train).detach().numpy()

# 画图对比
plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')
plt.plot(x_train.numpy(), predicted, label='Fitted line')
plt.legend()
plt.show()