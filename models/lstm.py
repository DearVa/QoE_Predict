import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, hidden_size, num_layers, batch_size, y_time_steps):
        super(Lstm, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # A_pkg_count A_pkg_bytes A_down_time V_pkg_count V_pkg_bytes V_down_time 以及之前的要预测的那个数据 7个输入
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # 注意这里指定batch_first为True
        # 设置了两个线性层
        self.linear1 = nn.Linear(in_features=hidden_size, out_features=int(hidden_size / 2))
        self.linear2 = nn.Linear(in_features=int(hidden_size / 2), out_features=y_time_steps)
        self.batch_size = batch_size
        self.y_time_steps = y_time_steps

    def forward(self, x):
        """
        x的size是batch_size * x_time_steps * 7
        :param x:
        :return:
        """
        h_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()
        c_0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).cuda()
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.linear1(out)
        out = self.linear2(out)
        # LSTM的最终的输出是y_time_steps个
        # out的size是 batch_size * x_time_steps * hidden_size
        # h和c的size是(num_directions * num_layers, batch_size, hidden_size)
        # 注意，不管有没有设置batch_first=True, batch_size永远在h和c的size的第二个
        # 而设置了batch_first=True之后，batch_size在output的size的第一个
        out = out[:, -self.y_time_steps, :]  # 只要最后y_time_step个cell的输出结果，out的size变为batch_size * y_time_steps * hidden_size
        # 经过整个网络之后，size由batch_size * x_time_steps * 7变成了batch_size * y_time_steps * self.y_time_steps
        return out
