# -*- coding: UTF-8 -*-
"""
@Time     : 2021/12/15 10:55
@Author   : Caiming Liu
@Version  : V1
@File     : CNN.py
@Software : PyCharm
"""

# Local Modules


# Third-party Modules
import torch.nn as nn
# Self-written Modules


class CNN_1D_Net(nn.Module):
    def __init__(self, out_channel=10):
        super(CNN_1D_Net, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4))

        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=512*4, out_features=256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256, out_features=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc(x)
        return x

