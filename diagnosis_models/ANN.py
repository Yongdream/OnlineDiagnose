# -*- coding: UTF-8 -*-
"""
@Time     : 2021/12/11 10:28
@Author   : Caiming Liu
@Version  : V1
@File     : ANN.py.py
@Software : PyCharm
"""

# Local Modules


# Third-party Modules
import torch
import torch.nn as nn
import warnings
# Self-written Modules

class ANN_Net(nn.Module):
    def __init__(self, args, out_channel=10):
        super(ANN_Net, self).__init__()

        if args.pretrained == True:
            warnings.warn("Pretrained model is not available")
        self.data_length = args.data_length
        if "FFT" in args.data_name:
            self.data_length = self.data_length // 2
        self.layer1 = nn.Sequential(
            nn.Linear(in_features=self.data_length, out_features=512),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(256, out_features=out_channel)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc(x)
        return x

