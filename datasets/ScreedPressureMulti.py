import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm


datasetname = ["fault",'healthy']


def get_files(root, op=3, signal_size=128):
    '''
    root: The location of the data set.
    '''
    data, lab = [], []

    for idx, name in enumerate(datasetname):
        data_dir = os.path.join(root, 'op_%d' % op, name)

        for item in os.listdir(data_dir):
            if item.endswith('.csv'):
                item_path = os.path.join(data_dir, item)
                data_load(item_path, idx, data, lab, signal_size)

    #print(data[0],len(lab))

    return data, lab


def data_load(item_path, label, data, lab, signal_size):
    '''
    This function is mainly used to generate test data and training data.
    '''
    fl = pd.read_csv(item_path,encoding='gbk',usecols=[0,1])
    #fl = fl.values.reshape(-1, 1)
    fl = fl.values

    start, end = 0, signal_size
    while end <= fl.shape[0]:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size


def data_transforms(dataset_type="train", normlize_type="-1-1"):
    transforms = {
        'train': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()

        ]),
        'val': Compose([
            Reshape(),
            Normalize(normlize_type),
            Retype()
        ])
    }
    return transforms[dataset_type]


class ScreedPressureMulti(object):
    num_classes = 2
    inputchannel = 2

    def __init__(self, data_dir='D:\IFD\data\ScreedPressure', normlizetype='mean-std', op=0, signal_size=128):
        self.data_dir = data_dir
        self.normlizetype = normlizetype
        self.op = int(op)
        self.signal_size=signal_size

    #根据工况读取数据集，不同时读取源域/目标域数据集
    def data_preprare(self, is_src=False):
        data, lab = get_files(self.data_dir, self.op, self.signal_size)
        data_pd = pd.DataFrame({"data": data, "label": lab})
        #data_pd = balance_data(data_pd)
        if is_src:
            train_dataset = dataset(list_data=data_pd, transform=data_transforms('train', self.normlizetype))
            return train_dataset
        else:
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=20, stratify=data_pd["label"])
            train_dataset = dataset(list_data=train_pd, transform=data_transforms('train', self.normlizetype))
            val_dataset = dataset(list_data=val_pd, transform=data_transforms('val', self.normlizetype))
            return train_dataset, val_dataset

#train_dataset, val_dataset=ScreedPressure().data_preprare()

