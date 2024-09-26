import os
from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets.SequenceDatasets import dataset
from datasets.sequence_aug import *
from tqdm import tqdm

dataname= {0:["TractionDrive_healthy_neg.csv","TractionDrive_fault_neg.csv"],#负极
           1:["TractionDrive_healthy_pos.csv","TractionDrive_fault_pos.csv"]}#正极

datasetname = ["TractionDriveHealthy","TractionDriveFault"]

label = [i for i in range(0, 2)]#牵引驱动健康,牵引驱动故障

# generate Training Dataset and Testing Dataset
def get_files(root, N, signal_size, signal_number):
    '''
    This function is used to generate the final training set and test set.
    root:The location of the data set
    '''

    data = []
    lab = []
    for k in range(len(N)):
        for n in tqdm(range(len(dataname[N[k]]))):
            if n == 0:
                path1 = os.path.join(root,datasetname[0], dataname[N[k]][n])  #datasetsClassFour/tractionSpeed/牵引驱动健康
            else:
                path1 = os.path.join(root,datasetname[1], dataname[N[k]][n])
            data1, lab1 = data_load(path1, label=label[n], signal_size=signal_size, signal_number=signal_number)
            data += data1
            lab += lab1

    return [data, lab]



def data_load(filename, label,signal_size, signal_number):
    '''
    This function is mainly used to generate test data and training data.
    filename:Data location
    '''
    fl = pd.read_csv(filename,encoding='gbk',usecols=[2])
    fl = fl.values.reshape(-1, 1)
    data = []
    lab = []
    start, end = 0, signal_size
    cnt = 0
    while end <= fl.shape[0] and cnt < signal_number:
        data.append(fl[start:end])
        lab.append(label)
        start += signal_size
        end += signal_size
        cnt += 1

    return data, lab

# --------------------------------------------------------------------------------------------------------------------
class TractionSpeed(object):
    num_classes = 2
    inputchannel = 1

    def __init__(self, data_dir, transfer_task, normlizetype="0-1", signal_size=512, signal_number=1000):
        self.data_dir = data_dir
        self.source_N = transfer_task[0]
        self.target_N = transfer_task[1]
        self.normlizetype = normlizetype
        self.signal_size = signal_size
        self.signal_number = signal_number
        self.data_transforms = {
            'train': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                # RandomAddGaussian(),
                # RandomScal    e(),
                # RandomStretch(),
                # RandomCrop(),
                Retype(),
                # Scale(1)
            ]),
            'val': Compose([
                Reshape(),
                Normalize(self.normlizetype),
                Retype(),
                # Scale(1)
            ])
        }

    def data_split(self, transfer_learning=True):
        if transfer_learning:
            # get source train
            list_data = get_files(self.data_dir, self.source_N, self.signal_size, self.signal_number)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            source_train = dataset(list_data=data_pd, transform=self.data_transforms['train'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N, self.signal_size, self.signal_number)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=20, stratify=data_pd["label"])
            target_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            target_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])
            return source_train, target_train, target_val
        else:
            #get source train and val
            list_data = get_files(self.data_dir, self.source_N, self.signal_size, self.signal_number)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            print(data_pd)
            train_pd, val_pd = train_test_split(data_pd, test_size=0.2, random_state=20, stratify=data_pd["label"])
            source_train = dataset(list_data=train_pd, transform=self.data_transforms['train'])
            source_val = dataset(list_data=val_pd, transform=self.data_transforms['val'])

            # get target train and val
            list_data = get_files(self.data_dir, self.target_N, self.signal_size, self.signal_number)
            data_pd = pd.DataFrame({"data": list_data[0], "label": list_data[1]})
            target_val = dataset(list_data=data_pd, transform=self.data_transforms['val'])
            return source_train, source_val, target_val