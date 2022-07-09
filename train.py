import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, dir_path, csv_file_names, x_time_steps, y_time_steps, stride, y_features):
        """
        初始化数据集
        :param dir_path: 所有csv文件的根目录
        :param csv_file_names: csv文件名列表（不含路径
        :param x_time_steps:
        :param y_time_steps:
        :param stride:
        """
        super(MyDataset, self).__init__()
        self.dir_path = dir_path
        self.csv_file_names = csv_file_names
        self.x_time_steps = x_time_steps
        self.y_time_steps = y_time_steps
        self.stride = stride
        self.y_features = y_features

    def __len__(self):
        return len(self.csv_file_names)

    @staticmethod
    def max_min_norm(array):
        min_val = array.min()
        max_min_delta = array.max() - min_val
        for i in range(len(array)):
            array[i] = (array[i] - min_val) / max_min_delta

    def __getitem__(self, idx):
        csv_path = os.path.join(self.dir_path, self.csv_file_names[idx])
        data = pd.read_csv(csv_path).iloc[:, 1:].values
        self.max_min_norm(data[:, 0])
        self.max_min_norm(data[:, 1])
        self.max_min_norm(data[:, 4])

        index = 0
        samples = []

        while (index + self.x_time_steps + self.y_time_steps) <= len(data) - 1:
            single_sample = data[index: index + self.x_time_steps + self.y_time_steps, :]
            samples.append(single_sample)
            # 每个single_sample的size是x_time_steps+y_time_steps*7
            # 前面的x_time_steps*7就是放入网络中的每个样本，后面的y_time_steps*7就是原始的true_y
            index += self.stride

        samples = torch.from_numpy(np.array(samples))

        x = samples[:, :-self.y_time_steps, :]
        if self.y_time_steps == 1:
            y = samples[:, -1, 2:].reshape(len(x), 1, self.y_features)
        else:
            y = samples[:, -self.y_time_steps:, :2]

        return x, y
