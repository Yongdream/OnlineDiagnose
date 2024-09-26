# -*- coding: UTF-8 -*-
"""
@Time     : 2021/12/16 21:32
@Author   : Caiming Liu
@Version  : V1
@File     : Bi-LSTM.py
@Software : PyCharm
"""

# Local Modules


# Third-party Modules
import torch
import torch.nn as nn

# Self-written Modules

class Bi_LSTM_Net(nn.Module):
    def __init__(self, args, out_channel=10):
        super(Bi_LSTM_Net, self).__init__()
        self.data_length = args.data_length
        self.step_length = args.step_length
        if "FFT" in args.data_name:
            self.data_length = self.data_length // 2
        self.layer1 = nn.Sequential(
            nn.LSTM(input_size=(self.data_length // self.step_length), hidden_size=512, num_layers=1, dropout=0.5, batch_first=True,
                    bidirectional=True),
        )

        self.fc = nn.Linear(in_features=512*2, out_features=out_channel)

    def forward(self, x):
        x = x.view(x.size(0), self.step_length, -1)
        x, _ = self.layer1(x)
        # batch_size, seq_len, hidden_size = x.shape
        x = x[:, -1, :]
        # print(x.shape)
        x = torch.as_tensor(x)
        x = self.fc(x)
        return x

