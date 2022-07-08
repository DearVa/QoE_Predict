import torch.nn as nn

# 设置超参数
x_time_steps = 5  # 用多少期去预测（在RNN模型中x_time_steps就是cell的个数）
y_time_steps = 1  # 老师说一般都只预测一期，所以y_time_steps应该就是固定值1，但是后面Env我懒得改了，所以这里还是保留了y_time_steps这个超参数
stride = 2  # 每次移动多少期来采样
hidden_size = 20
hidden_layers = 1

batch_size = 32
epochs = 200

# 搭建网络
class Lstm(nn.Module):
    def __init__(self, y_features):
        super(Lstm, self).__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=hidden_layers, batch_first=True)
        # 注意这里指定了batch_first为true
        # 这里设置了两个线性层，其实设置一层也可以
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2))
        self.linear2 = nn.Linear(in_features=int(hidden_size / 2), out_features=y_features)
        self.y_features = y_features

    def forward(self, x, h0):
        out, (h, c) = self.lstm(x)
        # x的size是batch_size*x_time_steps*x_features   本例中的x_features是7
        # LSTM的最终的输出是3个，h和c都是最后一个时刻的h、c
        # out的size是batch_size*x_time_steps*hidden_size
        # h和c 的size是(num_directions*num_layers,batch_size,hidden_size)。注意，不管有没有设置batch_first=True,batch_size永远在h和c的size的第二个。（
        # 而设置了batch_first=True之后，batch_size在output的size的第一个）
        out = out[:, -1, :]  # 只要最后一个cell的输出结果，out的size变为batch_size*1*hidden_size
        out = out.reshape(batch_size, -1)
        out = self.linear1(out)
        out = self.linear2(out).reshape(batch_size, 1, self.y_features)
        # 经过整个网络之后，size由batch_size*x_time_steps*x_features变成了batch_size*1*y_features
        return out
