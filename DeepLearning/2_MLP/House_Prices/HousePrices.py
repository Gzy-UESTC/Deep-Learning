import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1、读取数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(f"训练集规模: {train_data.shape}, 测试集规模: {test_data.shape}")

# 2、数据处理
# 合并训练集和测试集
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:])) 

# 找出所有数值类型的特征索引
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index

# 应用 Z-Score 标准化
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

# 缺失值填充为 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummy_na=True 会将缺失值也当作一个特征处理
all_features = pd.get_dummies(all_features, dummy_na=True)

# 转换回 PyTorch 张量
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
# 对价格取对数，这是因为 Kaggle 的评价指标是 RMSE(log(y), log(y_hat))
train_labels = torch.tensor(np.log(train_data.SalePrice.values.reshape(-1, 1)), dtype=torch.float32)

# 3、构建模型
class HousePriceModel(nn.Module):
    def __init__(self, in_features):
        super(HousePriceModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# 评价函数：对数均方根误差 (RMSE)
def get_rmse_log(net, features, labels):
    with torch.no_grad():
        preds = net(features)
        rmse = torch.sqrt(torch.mean((preds - labels)**2))
    return rmse.item()

# 4、训练流程
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    
    train_ls, test_ls = [], []
    dataset = TensorDataset(train_features, train_labels)
    train_iter = DataLoader(dataset, batch_size, shuffle=True)
    
    # 使用 Adam 优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss = nn.MSELoss()

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        
        train_ls.append(get_rmse_log(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(get_rmse_log(net, test_features, test_labels))
            
    return train_ls, test_ls

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.01, 0, 64

in_features = train_features.shape[1]
model = HousePriceModel(in_features)

train_ls, _ = train(model, train_features, train_labels, None, None,
                    num_epochs, lr, weight_decay, batch_size)

print(f'最终训练 log rmse: {train_ls[-1]:.4f}')