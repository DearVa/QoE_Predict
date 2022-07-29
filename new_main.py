import datetime
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lstm import Lstm, JsonLstm
from train import MyDataset, walk_all_file_paths, JsonDataset

from tensorboard_logger import configure, log_value

# videos = pd.read_csv(r'G:\Source\Python\AI\bupt\data\A\PCAP_FILES\baseline_Jan17_exp_42_videos.csv').iloc[:, :].values
# audios = pd.read_csv(r'G:\Source\Python\AI\bupt\data\A\PCAP_FILES\baseline_Jan17_exp_42_audios.csv').iloc[:, :].values
#
# plt.scatter(x=videos[:, :1], y=videos[:, 2:3], s=3, c='r')
# plt.scatter(x=audios[:, :1], y=audios[:, 2:3], s=1, c='b')
# plt.title('baseline_Jan17_exp_42')
# plt.show()


# 设置超参数
num_rows = 60

# 用10期x数据预测1期y数据
x_time_steps = 10
y_time_steps = 1

predict = 'rtt'  # jitter, rtt, fps

hidden_size = 128
num_layers = 8

if predict == 'rtt':
    # rtt需要特殊处理超参数，因为数量级不太一样
    batch_size = 100
    epochs = 1000

    learn_rate = 1e-4
    step_size = 5000
    gamma = 0.8
else:
    batch_size = 200
    epochs = 300

    learn_rate = 5e-4
    step_size = 1000
    gamma = 0.8

use_tensorboard = False

data_path = r'G:\Source\Python\AI\bupt\data_new'

json_paths = []
walk_all_file_paths('./data_new', '.json', json_paths)  # 遍历所有json文件的路径
random.shuffle(json_paths)

train_json_file_paths = json_paths[:int(len(json_paths) * 0.6)]
train_db = JsonDataset(train_json_file_paths, num_rows, x_time_steps, predict)

val_json_file_paths = json_paths[int(len(json_paths) * 0.6):int(len(json_paths) * 0.8)]
val_db = JsonDataset(val_json_file_paths, num_rows, x_time_steps, predict)

test_json_file_paths = json_paths[int(len(json_paths) * 0.8):]
test_db = JsonDataset(test_json_file_paths, num_rows, x_time_steps, predict)

train_loader = DataLoader(train_db, batch_size, drop_last=True)
val_loader = DataLoader(val_db, batch_size, drop_last=True)
test_loader = DataLoader(test_db, batch_size, drop_last=True)

model = JsonLstm(hidden_size, num_layers, batch_size, y_time_steps).cuda()

loss_fn = nn.MSELoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def r2_loss(output, target):
    """
    From https://en.wikipedia.org/wiki/Coefficient_of_determination
    """
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2

def evaluate_loss(loader, show_plt=False):
    """
    计算损失
    :param loader:
    :param show_plt: 显示图像
    :return:
    """
    loss_for_all_batch = []
    # r2_loss_for_all_batch = []
    for i, (x, y, f) in enumerate(loader):
        input_x, true_y = x.float().cuda(), y.float().cuda()
        with torch.no_grad():
            pre_y = model.forward(input_x)
            if show_plt and i % 10 == 0:
                plt.plot(true_y.cpu().detach().numpy(), c='r', alpha=0.75, label='truth')
                plt.plot(pre_y.cpu().detach().numpy(), c='b', alpha=0.75, label='pred')
                plt.title(f[0])
                plt.grid(axis='y')
                plt.legend()
                plt.show()
            loss_for_all_batch.append(loss_fn(pre_y, true_y).cpu())
            # r2_loss_for_all_batch.append(r2_loss(pre_y, true_y).cpu())
    return np.mean(loss_for_all_batch)  # , np.mean(r2_loss_for_all_batch)  # 用所有batch loss的均值代表该数据集上的总体loss水平


# model.load_state_dict(torch.load(r"G:\Source\Python\AI\bupt\2022-07-12_00-12-03_movement\20.pkl"))
# print('已将参数设置成训练过程中的最优值，现在开始测试test_set')
# loss_for_test = evaluate_loss(test_loader, True)
# print('测试集上的loss为：', loss_for_test)


def train():
    train_name = f'{predict[0]}_full_hs{hidden_size}_nl{num_layers}_bs{batch_size}_e{epochs}_lr{learn_rate}_ss{step_size}_g{gamma}_ts{x_time_steps}_ys{y_time_steps}'
    best_loss = 99999  # 因为希望val_loss不断减小，所以初始的val_loss设置大一点

    datetime_str = str(datetime.datetime.now().strftime('%H%M%S'))
    save_dir = r'.\models\{}_{}.pkl'.format(train_name, datetime_str)
    if use_tensorboard:
        configure("runs_new/{}".format(datetime_str), flush_secs=5)
    best_epoch = 0

    for it in range(epochs):
        for batch_index, data in enumerate(train_loader):
            x, y = data[0], data[1]
            input_x, true_y = x.float().cuda(), y.float().cuda()
            pre_y = model.forward(input_x)
            loss = loss_fn(pre_y, true_y)
            if batch_index == 0:
                print(loss.cpu().detach().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            # if (batch_index + 1) % 10 == 0:
            #     print('epoch：', it + 1, '   batch_index:', batch_index + 1, '  loss:', loss.item())

        # 每隔两个epoch就在val上看一下效果
        if it % 2 == 1:
            # loss
            loss = evaluate_loss(val_loader)
            if loss < best_loss:
                print('epoch: {} loss: {}↓'.format(it + 1, loss))
                best_epoch = it + 1
                best_loss = loss
                torch.save(model.state_dict(), save_dir)
            else:
                print('epoch: {} loss: {}'.format(it + 1, loss))

            if use_tensorboard:
                log_value('loss', loss, it)

    print('模型已训练完成，最好的epoch是{}，在验证集上的loss是{}'.format(best_epoch, best_loss))
    return save_dir

def test(save_dir: str):
    model.load_state_dict(torch.load(save_dir))
    print('开始测试test_set')
    loss = evaluate_loss(test_loader, True)
    print('测试集上的loss为：', loss)


if __name__ == '__main__':
    test(train())
    # test(r"G:\Source\Python\AI\bupt\models\f_full_hs128_nl8_bs200_e300_lr0.0005_ss1000_g0.8_ts10_ys1_122216.pkl")
