#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# import self define module
from . import file_operation
from . import functional


class ClsDataSetCreator(Dataset):
    """"""
    def __init__(self, data_dict: dict) -> None:
        self.data = torch.from_numpy(data_dict['data']).to(dtype=torch.float32)
        self.label = torch.from_numpy(data_dict['label']).to(dtype=torch.int64)
        self.n_samples = data_dict['label'].shape[0]

    def __getitem__(self, index):
        if len(self.data.shape) == 3:
            data = self.data[index, :, :]
            label = self.label[index]
        elif len(self.data.shape) == 2:
            data = self.data[index, :]
            label = self.label[index]
        else:
            raise ValueError('len(self.data.shape) Error.')
        return data, label

    def __len__(self):
        return self.n_samples

    @classmethod
    def creat_dataset(cls,
                      data_dict: dict,
                      bsz: int = 32,
                      is_shuffle: bool = True,
                      num_of_worker: int = 0,
                      ) -> (DataLoader, int):
        """"""
        assert 'data' in data_dict.keys()
        assert 'label' in data_dict.keys()

        data_set = cls(data_dict)
        batch_data_set = DataLoader(data_set, batch_size=bsz, shuffle=is_shuffle,
                                    num_workers=num_of_worker, pin_memory=True)
        return batch_data_set
