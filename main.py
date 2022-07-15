# 准备好数据
import datetime
import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from lstm import Lstm
from train import MyDataset

from tensorboard_logger import configure, log_value

# videos = pd.read_csv(r'G:\Source\Python\AI\bupt\data\A\PCAP_FILES\baseline_Jan17_exp_42_videos.csv').iloc[:, :].values
# audios = pd.read_csv(r'G:\Source\Python\AI\bupt\data\A\PCAP_FILES\baseline_Jan17_exp_42_audios.csv').iloc[:, :].values
#
# plt.scatter(x=videos[:, :1], y=videos[:, 2:3], s=3, c='r')
# plt.scatter(x=audios[:, :1], y=audios[:, 2:3], s=1, c='b')
# plt.title('baseline_Jan17_exp_42')
# plt.show()


# 设置超参数
num_rows = 121

# 用6期x数据预测1期y数据
x_time_steps = 6
predict = 'state'  # state, resolution, buf_health

hidden_size = 64
num_layers = 5

batch_size = 50
epochs = 100

use_tensorboard = False

base_path = r'G:\Source\Python\AI\bupt\data\A'

train_name = 'r_full_hs64_nl5_bs50_e100_lr1e-4_ss1000_g0.8_shu'

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


data_path = os.path.join(base_path, 'MERGED_FILES')
# csv_file_names = list(filter(lambda f: f.startswith('movement') and check_csv(f), os.listdir(data_path)))  # 遍历所有csv文件的路径
# csv_file_names = list(filter(lambda f: f.startswith('baseline') and check_csv(f), os.listdir(data_path)))  # 遍历所有csv文件的路径
csv_file_names = list(filter(check_csv, os.listdir(data_path)))  # 遍历所有csv文件的路径
random.shuffle(csv_file_names)

train_csvs_file_names = csv_file_names[:int(len(csv_file_names) * 0.6)]
train_db = MyDataset(data_path, train_csvs_file_names, num_rows, x_time_steps, predict)

val_csvs_file_names = csv_file_names[int(len(csv_file_names) * 0.6):int(len(csv_file_names) * 0.8)]
val_db = MyDataset(data_path, val_csvs_file_names, num_rows, x_time_steps, predict)

test_csvs_file_names = csv_file_names[int(len(csv_file_names) * 0.8):]
test_db = MyDataset(data_path, test_csvs_file_names, num_rows, x_time_steps, predict)

train_loader = DataLoader(train_db, batch_size, drop_last=True)
val_loader = DataLoader(val_db, batch_size, drop_last=True)
test_loader = DataLoader(test_db, batch_size, drop_last=True)

# 初始化模型、定义损失函数、优化器
if predict == 'state':
    y_time_steps = 4
elif predict == 'resolution':
    y_time_steps = 9
else:
    y_time_steps = 1

model = Lstm(hidden_size, num_layers, batch_size, y_time_steps).cuda()

if predict == 'state':
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
elif predict == 'resolution':
    loss_fn = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
else:
    loss_fn = nn.MSELoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

scheduler = lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.8)

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
    r2_loss_for_all_batch = []
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
            r2_loss_for_all_batch.append(r2_loss(pre_y, true_y).cpu())
    return np.mean(loss_for_all_batch), np.mean(r2_loss_for_all_batch)  # 用所有batch loss的均值代表该数据集上的总体loss水平


def evaluate_accuracy(loader, show_plt=False):
    """
    计算准确率
    :param loader:
    :return:
    """
    correct_count = 0
    error_count = 0
    for i, (x, y, f) in enumerate(loader):
        input_x = x.float().cuda()
        with torch.no_grad():
            pre_y = model.forward(input_x)
            if i == 0:
                print('truth: {} pre: {}'.format(int(y[0].cpu().item()), torch.argmax(pre_y[0]).cpu().item()))
            if show_plt:
                true_ys = []
                pre_ys = []

            length = len(pre_y)
            for j in range(length):
                true_y = y[j].int()
                cpu_pre_y = torch.argmax(pre_y[j]).cpu()
                if show_plt:
                    true_ys.append(true_y)
                    pre_ys.append(cpu_pre_y)

                if true_y == cpu_pre_y:
                    correct_count += 1
                else:
                    error_count += 1

                if show_plt and j == length - 1:
                    plt.plot(true_ys, c='r', alpha=0.75, label='truth')
                    plt.plot(pre_ys, c='b', alpha=0.75, label='pred')
                    plt.title(f[0])
                    plt.grid(axis='y')
                    plt.legend()
                    plt.show()
    return correct_count / (correct_count + error_count)


# model.load_state_dict(torch.load(r"G:\Source\Python\AI\bupt\2022-07-12_00-12-03_movement\20.pkl"))
# print('已将参数设置成训练过程中的最优值，现在开始测试test_set')
# loss_for_test = evaluate_loss(test_loader, True)
# print('测试集上的loss为：', loss_for_test)


def train():
    best_loss = 99999  # 因为希望val_loss不断减小，所以初始的val_loss设置大一点
    best_accuracy = 0

    datetime_str = str(datetime.datetime.now().strftime('%H%M%S'))
    save_dir = r'.\models\{}_{}.pkl'.format(train_name, datetime_str)
    if use_tensorboard:
        configure("runs/{}".format(datetime_str), flush_secs=5)
    best_epoch = 0

    for it in range(epochs):
        for batch_index, data in enumerate(train_loader):
            x, y = data[0], data[1]
            # if predict != 'buf_health':
            #     l = data[2]
            input_x, true_y = x.float().cuda(), y.long().squeeze().cuda()
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
            if predict == 'buf_health':
                # loss
                loss, r2 = evaluate_loss(val_loader)
                if loss < best_loss:
                    print('epoch: {} loss: {}↓ r2: {}'.format(it + 1, loss, r2))
                    best_epoch = it + 1
                    best_loss = loss
                    torch.save(model.state_dict(), save_dir)
                else:
                    print('epoch: {} loss: {} r2: {}'.format(it + 1, loss, r2))

                if use_tensorboard:
                    log_value('loss', loss, it)
            else:
                # accuracy
                accuracy = evaluate_accuracy(val_loader)
                if accuracy > best_accuracy:
                    print('epoch: {} accuracy: {}↑'.format(it + 1, accuracy))
                    best_epoch = it + 1
                    best_accuracy = accuracy
                    torch.save(model.state_dict(), save_dir)
                else:
                    print('epoch: {} accuracy: {}'.format(it + 1, accuracy))

                if use_tensorboard:
                    log_value('accuracy', accuracy, it)

    if predict == 'buf_health':
        print('模型已训练完成，最好的epoch是{}，在验证集上的loss是{}'.format(best_epoch, best_loss))
    else:
        print('模型已训练完成，最好的epoch是{}，在验证集上的accuracy是{}'.format(best_epoch, best_accuracy))

    return save_dir

def test(save_dir: str):
    model.load_state_dict(torch.load(save_dir))
    print('开始测试test_set')
    if predict == 'buf_health':
        loss = evaluate_loss(test_loader, True)
        print('测试集上的loss为：', loss)
    else:
        accuracy = evaluate_accuracy(test_loader, True)
        print('测试集上的accuracy为：', accuracy)


if __name__ == '__main__':
    test(train())
    # test(r"G:\Source\Python\AI\bupt\models\r_full_hs64_nl5_bs50_e100_lr1e-4_ss1000_g0.8_shu_214247.pkl")
