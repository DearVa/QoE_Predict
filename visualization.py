import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from main import check_csv

if __name__ == '__main__':
    data_path = r'G:\Source\Python\AI\bupt\data\A\MERGED_FILES'

    x = np.zeros(9)
    for file_name in list(filter(check_csv, os.listdir(data_path))):  # 遍历所有csv文件的路径
        data = pd.read_csv(os.path.join(data_path, file_name)).iloc[:, 2:].values  # 去掉前两列（index和timestamp）
        for res in data[:, 7]:
            x[int(res)] += 1

    plt.bar(range(9), x)
    plt.show()
