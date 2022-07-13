import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


loaded_datas = {}
"""用于存储已经加载过的数据，减少磁盘IO"""


class MyDataset(Dataset):
    def __init__(self, dir_path: str, csv_file_names: List[str], num_rows: int, x_time_steps: int, predict: str):
        """
        初始化数据集
        :param dir_path: 所有csv文件的根目录
        :param csv_file_names: csv文件名列表（不含路径
        :param num_rows: 每个csv文件的行数，一般是121
        :param x_time_steps: 每次抽的x数据条数
        :param predict: 预测的数据，可以为 'state' 'resolution' 'buf_health'
        """
        super(MyDataset, self).__init__()
        self.dir_path = dir_path
        self.csv_file_names = csv_file_names
        self.num_rows = num_rows
        self.x_time_steps = x_time_steps
        self.predict = predict

    def __len__(self):
        # 每个csv包含num_rows条数据
        # 抽样的时候忽略第0条
        # 所以一共有num_rows - 1条数据
        # 每次抽self.x_time_steps条数据
        # 所以一共有num_rows - 1 - self.x_time_steps + 1个样本
        return len(self.csv_file_names) * (self.num_rows - self.x_time_steps)

    @staticmethod
    def max_min_norm(array):
        min_val = array.min()
        max_min_delta = array.max() - min_val
        for i in range(len(array)):
            array[i] = (array[i] - min_val) / max_min_delta

    def __getitem__(self, idx: int):
        # 例如x_time_steps为20，就说明每次抽20条数据来预测
        n = self.num_rows - self.x_time_steps  # 每个csv能抽的数据总数
        i = idx // n  # csv文件索引
        j = (idx % n) + self.x_time_steps  # 抽的数据索引

        csv_file_name = self.csv_file_names[i]
        if csv_file_name not in loaded_datas:
            csv_path = os.path.join(self.dir_path, self.csv_file_names[i])
            data = pd.read_csv(csv_path).iloc[:, 2:].values  # 去掉前两列（index和timestamp）
            # data就是120行9列的数据，每行是一个时间点
            if np.size(data) != self.num_rows * 9:
                raise ValueError('data size error')

            # A_pkg_count A_pkg_bytes A_down_time V_pkg_count V_pkg_bytes V_down_time States Resolutions Buf_health

            # 归一化
            self.max_min_norm(data[:, 0])
            self.max_min_norm(data[:, 1])
            self.max_min_norm(data[:, 3])
            self.max_min_norm(data[:, 4])
            self.max_min_norm(data[:, 5])

            if self.predict == 'state':
                data[:, 6] += 1
            elif self.predict == 'buf_health':
                data[:, 8] /= 100

            loaded_datas[csv_file_name] = data
        else:
            data = loaded_datas[csv_file_name]

        # 先抽取y的数据，即States Resolutions Buf_health

        if self.predict == 'state':
            y_values = np.zeros(4, dtype=np.float32)
            label_index = int(data[j][6] + 1)
            y_values[label_index] = 1
        elif self.predict == 'resolution':
            y_values = np.zeros(9, dtype=np.float32)
            label_index = int(data[j][7])
            y_values[label_index] = 1
        else:
            y_values = np.array(data[j][8: 9], dtype=np.float32)

        # 再抽取x的数据，即A_pkg_count A_pkg_bytes A_down_time V_pkg_count V_pkg_bytes V_down_time
        # 从j - self.x_time_steps开始，抽取20行
        x_min = max(0, j - self.x_time_steps)
        x_max = max(0, j)
        x_values = np.array(data[x_min: x_max, :6], dtype=np.float32).reshape((x_max - x_min, 6))
        if self.predict == 'state':
            x_values = np.append(x_values, np.array(data[x_min: x_max, 6] / 4, dtype=np.float32).reshape(x_max - x_min, 1), axis=1)
        elif self.predict == 'resolution':
            x_values = np.append(x_values, np.array(data[x_min: x_max, 7], dtype=np.float32).reshape(x_max - x_min, 1), axis=1)
        else:
            x_values = np.append(x_values, np.array(data[x_min: x_max, 8], dtype=np.float32).reshape(x_max - x_min, 1), axis=1)
        # 左边不足的，补零
        # x_values = np.pad(x_values, ((self.x_time_steps - x_values.shape[0], 0), (0, 0)), 'constant', constant_values=0)

        if self.predict == 'buf_health':
            return torch.from_numpy(x_values), torch.from_numpy(y_values), csv_file_name
        else:
            return torch.from_numpy(x_values), torch.from_numpy(y_values), label_index, csv_file_name
