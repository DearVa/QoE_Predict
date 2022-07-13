import math
import os
from typing import List

import numpy as np
import pandas as pd
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
        buf_progress = reader.read_str()
        self.buf_progress = 0 if buf_progress == 'null' else float(buf_progress)
        self.buf_valid = reader.read_str() == 'true'

    def get_state(self):
        """
        获取状态，因为event的最后一位是“收集数据”，所以忽略
        :return: -1：无状态 0：缓冲 1：暂停 2：播放
        """
        for i in range(3):
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


class ChunkPack:
    """
    每隔5s集合一次数据
    """

    def __init__(self, video_csv_path, audio_csv_path):
        videos = pd.read_csv(video_csv_path).iloc[:].values
        audios = pd.read_csv(audio_csv_path).iloc[:].values

        start_time = min(videos[0][4], audios[0][4]) / 1000.0
        end_time = 0
        for video in videos:
            end_time = max(end_time, video[4] / 1000.0 + video[3])
        for audio in audios:
            end_time = max(end_time, audio[4] / 1000.0 + audio[3])

        length = math.ceil((end_time - start_time) / 5)

        self.dict = {
            "a_pkg_count": np.zeros(length),
            "a_pkg_bytes": np.zeros(length),
            "a_down_time": np.zeros(length),
            "v_pkg_count": np.zeros(length),
            "v_pkg_bytes": np.zeros(length),
            "v_down_time": np.zeros(length),
        }

        for i in range(len(videos)):
            down_start = videos[i][4] / 1000.0
            down_time = videos[i][3]
            index = int((down_start - start_time) / 5)
            end_index = int(((down_start + down_time) - start_time) / 5)
            while index <= end_index:
                time = min(down_start + down_time, (index + 1) * 5.0 + start_time)
                part = min(1.0, (time - max(down_start, index * 5.0 + start_time)) / down_time)
                self.dict["v_pkg_count"][index] += videos[i][1] * part
                self.dict["v_pkg_bytes"][index] += videos[i][2] * videos[i][1] * part
                self.dict["v_down_time"][index] += down_time * part
                index += 1

        for i in range(len(audios)):
            down_start = audios[i][4] / 1000.0
            down_time = audios[i][3]
            index = int((down_start - start_time) / 5)
            end_index = int(((down_start + down_time) - start_time) / 5)
            while index <= end_index:
                time = min(down_start + down_time, (index + 1) * 5.0 + start_time)
                part = min(1.0, (time - max(down_start, index * 5.0 + start_time)) / down_time)
                self.dict["a_pkg_count"][index] += audios[i][1] * part
                self.dict["a_pkg_bytes"][index] += audios[i][2] * part
                self.dict["a_down_time"][index] += down_time * part
                index += 1

    def __getitem__(self, idx):
        return self.dict[idx]


class DataPack:
    """
    每隔5s集合一次数据
    """

    def __init__(self, datas: List[Data], chunk_pack: ChunkPack):
        self.dict = {
            "timestamp": [],
            # "bytes_recvs": [],
            # "pkg_bytes": [],

            "a_pkg_count": chunk_pack["a_pkg_count"],
            "a_pkg_bytes": chunk_pack["a_pkg_bytes"],
            "a_down_time": chunk_pack["a_down_time"],
            "v_pkg_count": chunk_pack["v_pkg_count"],
            "v_pkg_bytes": chunk_pack["v_pkg_bytes"],
            "v_down_time": chunk_pack["v_down_time"],

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
                self.dict["timestamp"].append(data.playback_info.epoch_time)
                self.dict["resolutions"].append(data.playback_info.resolution)
                self.dict["buf_healths"].append(data.playback_info.buf_health)
            n -= 1
            if n == 0:
                n = 50
                # self.dict["bytes_recvs"].append(temp_pkg[0])
                # self.dict["pkg_bytes"].append(temp_pkg[1])
                self.dict["states"].append(temp_pkg[2])
                temp_pkg = [0, 0, -1]

        # self.dict["bytes_recvs"].append(temp_pkg[0])
        # self.dict["pkg_bytes"].append(temp_pkg[1])
        self.dict["states"].append(temp_pkg[2])

        max_length = max(list(map(len, self.dict.values())))
        for k, v in self.dict.items():
            self.dict[k] = np.pad(np.array(v), (0, max_length - len(v)), 'constant', constant_values=(0, 0))

    @staticmethod
    def show_plt_internal(i: int, datas: List, title: str):
        plt.subplot(5, 1, i)
        plt.scatter(x=range(len(datas)), y=datas, s=1)
        plt.title(title, y=0.5, loc='right')

    def show_plt(self, title):
        for i, (k, v) in enumerate(self.dict.items()):
            self.show_plt_internal(i + 1, v, k)
            if i == 0:
                plt.title(title, y=1)
        plt.show()

    def save_fig(self, save_path):
        for i, (k, v) in enumerate(self.dict.items()):
            self.show_plt_internal(i + 1, v, k)
            if i == 0:
                plt.title(os.path.split(save_path)[1][:-4], y=1)
        plt.savefig(save_path)
        plt.show()

    def save(self, save_path):
        pd.DataFrame(self.dict).to_csv(save_path)

    def __getitem__(self, idx):
        return self.dict[idx]


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


# chunks = pd.read_csv(r"G:\Source\Python\AI\bupt\data\A\PCAP_FILES\baseline_Jan17_exp_28_videos.csv").iloc[:].values
# start_time = chunks[0][4]
# for chunk in chunks:
#     down_start = (chunk[4] - start_time) / 1000
#     xs = [down_start, down_start + chunk[3]]
#     ys = [chunk[1], chunk[1]]
#     plt.plot(xs, ys)
# plt.grid(axis='x')
# plt.xlim((0, 40))
# plt.show()

# chunks = pd.read_csv(r"G:\Source\Python\AI\bupt\data\A\MERGED_FILES\baseline_Jan17_exp_30_merged.csv").iloc[:].values
# start_time = chunks[1][1]
# for i, chunk in enumerate(chunks):
#     if i == 0:
#         continue
#     xs = [i * 5, i * 5 + chunk[4]]
#     ys = [chunk[2], chunk[2]]
#     plt.plot(xs, ys)
# plt.grid(axis='x')
# plt.xlim((0, 40))
# plt.show()

if __name__ == '__main__':
    path = './data/A'
    merged_path = os.path.join(path, 'MERGED_FILES')

    for r, d, file_names in os.walk(merged_path):
        for file_name in file_names:
            file_path = os.path.join(merged_path, file_name)
            # if os.path.exists(file_path[:-3] + 'csv'):
            #     continue
            if file_name.endswith('txt'):
                video_csv_path = os.path.join(path, 'PCAP_FILES', file_name[:-10] + 'videos.csv')
                if not os.path.exists(video_csv_path):
                    continue
                audio_csv_path = os.path.join(path, 'PCAP_FILES', file_name[:-10] + 'audios.csv')
                if not os.path.exists(audio_csv_path):
                    continue

                chunk_pack = ChunkPack(video_csv_path, audio_csv_path)

                with open(file_path) as f:
                    datas = []
                    try:
                        for line in f:
                            if len(line) > 10:  # 排除空行
                                datas.append(Data(line))

                        train_data = DataPack(datas, chunk_pack)
                        # train_data.save_fig(file_path[:-3] + 'jpg')
                        train_data.save(file_path[:-3] + 'csv')
                        print(file_name)
                    except:
                        print('error: ' + file_name)

    print('fin')
