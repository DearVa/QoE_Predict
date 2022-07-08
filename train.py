# 创建数据集    
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

y_features = 2


class Env(Dataset):
    def __init__(self, root, x_time_steps, y_time_steps, stride, mode):
        super(Env, self).__init__()
        self.data = pd.read_excel(root).iloc[:, :-1].values
        self.x_time_steps = x_time_steps
        self.y_time_steps = y_time_steps
        self.stride = stride
        self.mode = mode
        self.samples = self.creat_xy('./final_sample.pkl')

        self.x = self.samples[:, :-self.y_time_steps, :]
        if self.y_time_steps == 1:
            self.y = self.samples[:, -1, the_col_wanted].reshape(len(self.x), 1, y_features)
        else:
            self.y = self.samples[:, -self.y_time_steps:, the_col_wanted]

        if self.mode == 'train':
            self.x = self.x[:int(0.6 * len(self.x)), :, :]
            self.y = self.y[:int(0.6 * len(self.y)), :, :]

        if self.mode == 'val':
            self.x = self.x[int(0.6 * len(self.x)):int(0.8 * len(self.x)), :, :]
            self.y = self.y[int(0.6 * len(self.y)):int(0.8 * len(self.y)), :, :]

        if self.mode == 'test':
            self.x = self.x[int(0.8 * len(self.x)):, :, :]
            self.y = self.y[int(0.8 * len(self.y)):, :, :]

    def creat_xy(self, save_path):
        # 此函数用于创造sample，每个样本的size是x_time_steps+y_time_steps*7
        # 前面的x_time_steps*7就是放入网络中的每个样本，后面的y_time_steps*7就是原始的true_y
        index = 0
        samples = []

        while (index + self.x_time_steps + self.y_time_steps) <= (len(self.data) - 1):
            single_sample = self.data[index: index + self.x_time_steps + self.y_time_steps, :]
            samples.append(single_sample)
            # 每个single_sample的size是x_time_steps+y_time_steps*7
            # 前面的x_time_steps*7就是放入网络中的每个样本，后面的y_time_steps*7就是原始的true_y
            index += self.stride
        else:
            final_sample = torch.from_numpy(np.array(samples))
            with open(save_path, 'wb') as f:  # 将数据写入pkl文件
                pickle.dump(final_sample, f)
            return final_sample

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx, :, :], self.y[idx, :, :]
        return x, y
