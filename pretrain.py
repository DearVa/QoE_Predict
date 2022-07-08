import os
from typing import List

from matplotlib import pyplot as plt


class LineReader:
    def __init__(self, line: str):
        self.line = line
        self.i = 0

    def read_str(self):
        result = ''
        while self.i < len(self.line) and self.line[self.i] != ',':
            if self.line[self.i] != '[' and self.line[self.i] != ']' and self.line[self.i] != ' ':
                result += self.line[self.i]
            self.i += 1
        self.i += 1
        return result

    def read_int(self):
        return int(self.read_str())

    def read_float(self):
        return float(self.read_str())

    def read_bool(self):
        return self.read_str() == '1'


class NetInfo:
    def __init__(self, reader: LineReader):
        self.ip_src = reader.read_str()
        self.ip_dst = reader.read_str()
        self.protocol = reader.read_str()
        self.pkg_send = reader.read_int()
        self.pkg_recv = reader.read_int()
        self.bytes_send = reader.read_int()
        self.bytes_recv = reader.read_int()


class PlaybackInfo:
    def __init__(self, reader: LineReader):
        # “缓冲”、“暂停”、“播放”和“收集数据”
        self.event = (reader.read_bool(), reader.read_bool(), reader.read_bool(), reader.read_bool())
        self.epoch_time = reader.read_int()
        self.start_time = reader.read_int()
        self.progress = reader.read_float()
        self.length = reader.read_float()
        self.quality = (reader.read_bool(), reader.read_bool(), reader.read_bool(),
                        reader.read_bool(), reader.read_bool(), reader.read_bool(),
                        reader.read_bool(), reader.read_bool(), reader.read_bool())

        self.resolution = 0
        for i in range(9):
            if self.quality[-i]:
                self.resolution = i
                break
        self.buf_health = reader.read_float()
        self.buf_progress = reader.read_float()
        self.buf_valid = reader.read_str() == 'true'

    def get_state(self):
        """
        获取状态，因为event的最后一位是“收集数据”，所以忽略
        :return: -1：无状态 0：缓冲 1：暂停 2：播放
        """
        for i in range(4):
            if self.event[i]:
                return i
        return -1


class Data:
    def __init__(self, line: str):
        reader = LineReader(line)
        self.time = reader.read_float()
        self.pkg_send = reader.read_int()
        self.pkg_recv = reader.read_int()
        self.bytes_send = reader.read_int()
        self.bytes_recv = reader.read_int()
        self.net_infos = []
        for i in range(26):
            self.net_infos.append(NetInfo(reader))
        self.playback_info = PlaybackInfo(reader)


def normalize(array):
    _min = min(array)
    _range = max(array) - _min
    for i in range(len(array)):
        array[i] = (array[i] - _min) / _range
    return array


class DataPack:
    """
    每隔5s集合一次数据
    """

    def __init__(self, datas: List[Data]):
        self.dict = {
            "bytes_recvs": [],
            "pkg_bytes": [],
            "states": [],
            "resolutions": [],
            "buf_healths": []
        }

        n = 50
        temp_pkg = [0, 0, -1]
        for data in datas:
            temp_pkg[0] += data.bytes_recv
            temp_pkg[1] += 0 if data.pkg_recv == 0 else data.bytes_recv / data.pkg_recv
            state = data.playback_info.get_state()
            if state != -1 and temp_pkg[2] == -1:  # 获取第一个有效状态
                temp_pkg[2] = state
            if n == 50:
                self.dict["resolutions"].append(data.playback_info.resolution)
                self.dict["buf_healths"].append(data.playback_info.buf_health)
            n -= 1
            if n == 0:
                n = 50
                self.dict["bytes_recvs"].append(temp_pkg[0])
                self.dict["pkg_bytes"].append(temp_pkg[1])
                self.dict["states"].append(temp_pkg[2])
                temp_pkg = [0, 0, -1]

        self.dict["bytes_recvs"].append(temp_pkg[0])
        self.dict["pkg_bytes"].append(temp_pkg[1])
        self.dict["states"].append(temp_pkg[2])

    @staticmethod
    def show_plt_internal(i: int, datas: List, title: str):
        plt.subplot(5, 1, i)
        plt.scatter(x=range(len(datas)), y=datas, s=1)
        plt.title(title, y=0.5, loc='right')

    def show_plt(self):
        i = 1
        for k, v in self.dict.items():
            self.show_plt_internal(i, v, k)
            i += 1
        plt.show()


class TrainData1:
    """
    归一化之后的Data，包括pkg_recvs，bytes_recv，resolution和buf_health
    """

    def __init__(self, datas: List[Data]):
        self.bytes_recvs = list(map(lambda d: d.bytes_recv, datas))
        self.pkg_bytes = list(map(lambda d: 0 if d.pkg_recv == 0 else d.bytes_recv / d.pkg_recv, datas))
        self.resolutions = list(map(lambda d: d.playback_info.resolution, datas))
        self.buf_healths = list(map(lambda d: d.playback_info.buf_health, datas))

    @staticmethod
    def show_plt_internal(i: int, datas: List, title: str):
        plt.subplot(4, 1, i)
        plt.plot(datas)
        plt.title(title, y=0.5, loc='right')

    def show_plt(self):
        self.show_plt_internal(2, self.bytes_recvs, 'bytes_recvs')
        self.show_plt_internal(1, self.pkg_bytes, 'pkg_bytes')
        self.show_plt_internal(3, self.resolutions, 'resolutions')
        self.show_plt_internal(4, self.buf_healths, 'buf_healths')
        plt.show()


for r, d, files in os.walk('./data/A/MERGED_FILES'):
    for file in files:
        if file.endswith('txt'):
            with open(file) as f:
                datas = []
                for line in f:
                    if len(line) > 10:  # 排除空行
                        datas.append(Data(line))

                train_data = DataPack(datas)
                train_data.show_plt()
