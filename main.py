from pretrain import Data, TrainData1, DataPack
import matplotlib.pyplot as plt

with open('./data/A/MERGED_FILES/baseline_Jan17_exp_33_merged.txt') as f:
    datas = []
    for line in f:
        if len(line) > 10:  # 随便设个长度
            datas.append(Data(line))

# plt.subplot(6, 1, 1)
# plt.plot(list(map(lambda d: d.pkg_send, datas)))
# plt.title('pkg_send', y=0.5, loc='right')
#
# plt.subplot(6, 1, 2)
# plt.plot(list(map(lambda d: d.pkg_recv, datas)))
# plt.title('pkg_recv', y=0.5, loc='right')
#
# plt.subplot(6, 1, 3)
# plt.plot(list(map(lambda d: d.bytes_send, datas)))
# plt.title('bytes_send', y=0.5, loc='right')
#
# plt.subplot(6, 1, 4)
# plt.plot(list(map(lambda d: d.bytes_recv, datas)))
# plt.title('bytes_recv', y=0.5, loc='right')
#
# plt.subplot(6, 1, 5)
# plt.plot(list(map(lambda d: d.playback_info.buf_health, datas)))
# plt.title('buf_health', y=0.5, loc='right')
#
# plt.subplot(6, 1, 6)
# plt.plot(list(map(lambda d: d.playback_info.resolution, datas)))
# plt.title('resolution', y=0.5, loc='right')

train_data = DataPack(datas)
train_data.show_plt()
