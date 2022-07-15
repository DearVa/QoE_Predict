import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
        self.num_rows = num_rows
        self.x_time_steps = x_time_steps
        self.predict = predict
        self.loaded_datas = []

        for csv_file_name in csv_file_names:
            csv_path = os.path.join(self.dir_path, csv_file_name)
            data = pd.read_csv(csv_path).iloc[:, 2:].values  # 去掉前两列（index和timestamp）
            # data就是120行9列的数据，每行是一个时间点
            if np.size(data) != self.num_rows * 9:
                raise ValueError('data size error')

            # A_pkg_count A_pkg_bytes A_down_time V_pkg_count V_pkg_bytes V_down_time States Resolutions Buf_health

            # 归一化
            self.max_min_norm(data[:, 0])
            self.max_min_norm(data[:, 1])
            self.max_min_norm(data[:, 2])
            self.max_min_norm(data[:, 3])
            self.max_min_norm(data[:, 4])
            self.max_min_norm(data[:, 5])

            if self.predict == 'state':
                data[:, 6] += 1
            elif self.predict == 'buf_health':
                data[:, 8] /= 100

            for i in range(x_time_steps, num_rows - x_time_steps):
                # 先抽取y的数据，即States Resolutions Buf_health

                if self.predict == 'state':
                    # y_values = np.zeros(4, dtype=np.float32)
                    # label_index = int(data[i][6] + 1)
                    # y_values[label_index] = 1
                    y_values = np.array(data[i][6: 7], dtype=np.float32)
                elif self.predict == 'resolution':
                    # y_values = np.zeros(9, dtype=np.float32)
                    # label_index = int(data[i][7])
                    # y_values[label_index] = 1
                    y_values = np.array(data[i][7: 8], dtype=np.float32)
                else:
                    y_values = np.array(data[i][8: 9], dtype=np.float32)

                # 再抽取x的数据，即A_pkg_count A_pkg_bytes A_down_time V_pkg_count V_pkg_bytes V_down_time
                # 从i - self.x_time_steps开始，抽取x_time_steps行
                x_min = max(0, i - self.x_time_steps)
                x_max = max(0, i)

                x_values = np.array(data[x_min: x_max, :6], dtype=np.float32)
                if not np.any(x_values):
                    continue

                dx = x_max - x_min
                x_values = x_values.reshape((dx, 6))
                if self.predict == 'state':
                    append_data = np.zeros((dx, 4), dtype=np.float32)
                    for j in range(dx):
                        append_data[j][int(data[x_min + j][6])] = 1
                    x_values = np.append(x_values, append_data, axis=1)
                elif self.predict == 'resolution':
                    append_data = np.zeros((dx, 9), dtype=np.float32)
                    for j in range(dx):
                        append_data[j][int(data[x_min + j][7])] = 1
                    x_values = np.append(x_values, append_data, axis=1)
                else:
                    x_values = np.append(x_values, np.array(data[x_min: x_max, 8], dtype=np.float32).reshape(dx, 1), axis=1)
                # 左边不足的，补零
                # x_values = np.pad(x_values, ((self.x_time_steps - x_values.shape[0], 0), (0, 0)), 'constant', constant_values=0)

                self.loaded_datas.append((torch.from_numpy(x_values), torch.from_numpy(y_values), csv_file_name))

    def __len__(self):
        return len(self.loaded_datas)

    @staticmethod
    def max_min_norm(array):
        min_val = array.min()
        max_min_delta = array.max() - min_val
        for i in range(len(array)):
            array[i] = (array[i] - min_val) / max_min_delta

    def __getitem__(self, idx: int):
        return self.loaded_datas[idx]
