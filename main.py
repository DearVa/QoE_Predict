# 准备好数据
import os
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models.lstm import Lstm
from train import MyDataset

# 设置超参数
x_time_steps = 5  # 用多少期去预测（在RNN模型中x_time_steps就是cell的个数）
y_time_steps = 1  # 预测多少期
stride = 2  # 每次移动多少期来采样
hidden_size = 20
hidden_layers = 1
y_features = 3  # 预测3个特征

batch_size = 32
epochs = 200

base_path = r'G:\Source\Python\AI\bupt\data\A'


def check_csv(csv_file_name: str):
    if not csv_file_name.endswith('.csv'):
        return False
    video_csv_path = os.path.join(base_path, 'PCAP_FILES', csv_file_name[:-10] + 'videos.csv')
    if not os.path.exists(video_csv_path):
        return False
    audio_csv_path = os.path.join(base_path, 'PCAP_FILES', csv_file_name[:-10] + 'audios.csv')
    if not os.path.exists(audio_csv_path):
        return False
    return True


csv_file_names = list(filter(check_csv, os.listdir(os.path.join(base_path, 'MERGED_FILES'))))  # 遍历所有csv文件的路径
random.shuffle(csv_file_names)

train_csvs_file_names = csv_file_names[:int(len(csv_file_names) * 0.6)]
train_db = MyDataset(base_path, train_csvs_file_names, x_time_steps, y_time_steps, stride, y_features)

val_csvs_file_names = csv_file_names[int(len(csv_file_names) * 0.6):int(len(csv_file_names) * 0.8)]
val_db = MyDataset(base_path, val_csvs_file_names, x_time_steps, y_time_steps, stride, y_features)

test_csvs_file_names = csv_file_names[int(len(csv_file_names) * 0.8):]
test_db = MyDataset(base_path, test_csvs_file_names, x_time_steps, y_time_steps, stride, y_features)

train_loader = DataLoader(train_db, batch_size, drop_last=True)
val_loader = DataLoader(val_db, batch_size, drop_last=True)
test_loader = DataLoader(test_db, batch_size, drop_last=True)

# 初始化模型、定义损失函数、优化器
model = Lstm(hidden_size, hidden_layers, batch_size, y_features)
h0, c0 = torch.zeros([hidden_layers, batch_size, hidden_size]), torch.zeros([hidden_layers, batch_size, hidden_size])
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
best_loss = 99999  # 因为希望val_loss不断减小，所以初始的val_loss设置大一点


# 设置一个evaluate函数，用于评估模型的效果(这里使用loss来衡量，根据实际情况，也可以选择precision、recall、F_β score、auc等来评估)
def evaluate(loader):
    loss_for_all_batch = []
    for (x, y) in loader:
        input_x = x.float()
        true_y = y.float()
        with torch.no_grad():
            pre_y = model.forward(input_x, (h0, c0))
            loss = loss_fn(pre_y, true_y)  # 每个batch的loss
            loss_for_all_batch.append(loss)
    loss_for_this_loader = np.mean(loss_for_all_batch)  # 用所有batch loss的均值代表该数据集上的总体loss水平
    return loss_for_this_loader


# 开始训练
for it in range(epochs):
    for batch_index, (x, y) in enumerate(train_loader):
        input_x = x.float()
        true_y = y.float()
        pre_y = model.forward(input_x, (h0, c0))
        loss = loss_fn(pre_y, true_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_index + 1) % 10 == 0:
            print('epoch：', it + 1, '   batch_index:', batch_index + 1, '  loss:', loss.item())

    # 每隔两个epoch就在val上看一下效果
    if it % 2 == 1:
        loss_for_val = evaluate(val_loader)
        if loss_for_val < best_loss:
            print('已经完 成了{}次迭代，val的loss有所下降,val_loss为：{}'.format(it + 1, loss_for_val))
            best_epoch = it + 1
            best_loss = loss_for_val
            torch.save(model.state_dict(), 'best_model_ckp.txt')

print('模型已训练完成，最好的epoch是{}，在验证集上的loss是{}'.format(best_epoch, best_loss))

model.load_state_dict(torch.load('best_model_ckp.txt'))
print('已将参数设置成训练过程中的最优值，现在开始测试test_set')
loss_for_test = evaluate(test_loader)
print('测试集上的loss为：', loss_for_test)
