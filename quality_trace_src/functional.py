#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
""""""
import sys

import numpy as np
import os
import math

import pandas as pd
import torch
import datetime
import random
from . import file_operation


def cut_sequence(in_data: np.ndarray, out_len: int = None) -> np.ndarray:
    assert len(in_data.shape) == 1
    assert in_data.shape[-1] > out_len
    out_data = in_data[0: out_len]
    return out_data


def median(data: list) -> float:
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half])/2


def get_rnd_file_list(path: str, rate: float) -> list:
    """"""
    file_list = os.listdir(path)
    n_files = len(file_list)
    n_res = math.floor(n_files * rate)
    random.shuffle(file_list)
    temp_rnd_list = file_list[0:n_res]
    return temp_rnd_list


def sequenceDiff(in_sq):
    """计算差分序列"""
    sq_1 = in_sq[0:-1]
    sq_2 = in_sq[1:]
    sq_diff = sq_2 - sq_1
    return sq_diff


def round_precision(x, precision=0):
    """精确四舍五入"""
    val = x * 10**precision
    int_part = math.modf(val)[1]
    fractional_part = math.modf(val)[0]
    out = 0
    if fractional_part >= 0.5:
        out = int_part + 1
    else:
        out = int_part
    out = out / 10**precision
    return out


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


class Tokenizer():
    """"""
    def __init__(self, token_tuple=(32, True, 16)):
        """
        token_tup = (t_len, overlap, step)
        """
        self.t_len = token_tuple[0]
        self.overlap = token_tuple[1]
        self.step = token_tuple[2]
        self.detoken_len = None

    def calculate_expamle_detoken_len(self, example_path):
        """"""
        data_list = os.listdir(example_path)
        data_path = os.path.join(example_path, data_list[0])
        in_data = np.load(data_path)[0:-1]  # data specialize
        in_temp = in_data
        d_size = in_temp.shape
        r_mod = d_size[0] % self.t_len
        if not self.overlap:
            num_of_padding = 0
            if r_mod != 0:
                pad_num = in_temp[-1]
                num_of_padding = self.t_len - r_mod
            de_token_len = d_size[0] + num_of_padding
        else:
            num_of_step = math.ceil((d_size[0] - (self.t_len - self.step)) / self.step)
            de_token_len = (num_of_step - 1) * self.step + self.t_len
        self.detoken_len = de_token_len


    def tokenize(self, in_data):
        """"""
        in_temp = in_data
        d_size = in_temp.shape
        assert d_size[0] > self.t_len
        r_mod = d_size[0] % self.t_len

        if not self.overlap:
            if r_mod != 0:
                pad_num = in_temp[-1]
                num_of_padding = self.t_len - r_mod
                pad_arr = np.ones(num_of_padding) * pad_num
                in_temp = np.concatenate((in_temp, pad_arr))
            out_data = np.reshape(in_temp, (-1, self.t_len))
            num_of_token = out_data.shape[0]
        else:
            num_of_step = math.ceil((d_size[0] - (self.t_len - self.step)) / self.step)
            detoken_len = (num_of_step - 1) * self.step + self.t_len
            if (detoken_len % d_size[0]) != 0:
                pad_num = in_temp[-1]
                num_of_padding = detoken_len - d_size[0]
                pad_arr = np.ones(num_of_padding) * pad_num
                in_temp = np.concatenate((in_temp, pad_arr))
            # overlap tokenize
            out_data = np.zeros((num_of_step, self.t_len))
            for stp in range(num_of_step):
                index = stp * self.step
                temp_token = in_temp[index:index + self.t_len]
                out_data[stp, :] = temp_token
            num_of_token = out_data.shape[0]
        return out_data

    def detokenize(self, in_data):
        """"""
        org_size = in_data.shape
        if not self.overlap:
            out_data = in_data.view(1, -1)
        else:
            num_of_token = org_size[0]
            out_data = torch.zeros((num_of_token - 1) * self.step + self.t_len)
            first_token = in_data[0, :]
            out_data[0:self.t_len] = first_token  # put first token into out sequence
            for i in range(1, num_of_token):
                curr_token = in_data[i, :]  # get token from second token
                curr_start_index = i * self.step
                curr_end_index = curr_start_index + self.t_len
                padded_curr_token = torch.zeros((num_of_token - 1) * self.step + self.t_len)
                padded_curr_token[curr_start_index: curr_end_index] = curr_token
                out_data += padded_curr_token
                curr_mid_start_index = curr_start_index
                curr_mid_end_index = curr_start_index + self.step
                out_data[curr_mid_start_index: curr_mid_end_index] /= 2
        return out_data

    def token_wrapper(self, data, *args):
        """"""
        if args[0] == 'token':
            assert (len(data.shape) == 1) and (type(data) is np.ndarray)
            arr_token = self.tokenize(data)
        elif args[0] == 'detoken':
            # in_data is a tensor:(number of token, token length)
            assert torch.is_tensor(data) and (len(data.shape) == 2)
            arr_token = self.detokenize(data)
        else:
            raise Exception('Tokenize Mode Error.')
        # convert data
        re_data = arr_token
        return re_data


def tokenize(in_data, t_len, is_overlap, step):
    """"""
    in_temp = in_data
    d_size = in_temp.shape
    r_mod = d_size[0] % t_len
    if not is_overlap:
        num_of_padding = 0
        if r_mod != 0:
            pad_num = in_temp[-1]
            num_of_padding = t_len - r_mod
            pad_arr = np.ones(num_of_padding) * pad_num
            in_temp = np.concatenate((in_temp, pad_arr))
        out_data = np.reshape(in_temp, (-1, t_len))
        num_of_token = out_data.shape[0]
        detoken_len = d_size[0] + num_of_padding
    else:
        num_of_step = math.ceil((d_size[0] - (t_len - step)) / step)
        detoken_len = (num_of_step - 1) * step + t_len
        if (detoken_len % d_size[0]) != 0:
            pad_num = in_temp[-1]
            num_of_padding = detoken_len - d_size[0]
            pad_arr = np.ones(num_of_padding) * pad_num
            in_temp = np.concatenate((in_temp, pad_arr))
        # overlap tokenize
        out_data = np.zeros((num_of_step, t_len))
        for stp in range(num_of_step):
            index = stp * step
            temp_token = in_temp[index:index + t_len]
            out_data[stp, :] = temp_token
        num_of_token = out_data.shape[0]
    return out_data


def cal_detoken_len(in_len: int, t_len: int, is_overlap: bool, step: int) -> int:
    """"""
    r_mod = in_len % t_len
    if not is_overlap:
        num_of_padding = 0
        if r_mod != 0:
            num_of_padding = t_len - r_mod
        detoken_len = in_len + num_of_padding
    else:
        num_of_step = math.ceil((in_len - (t_len - step)) / step)
        detoken_len = (num_of_step - 1) * step + t_len
    return detoken_len


def cal_n_token(in_len: int, t_len: int, is_overlap: bool, step: int) -> int:
    """"""
    r_mod = in_len % t_len
    if not is_overlap:
        num_of_padding = 0
        if r_mod != 0:
            num_of_padding = t_len - r_mod
        n_token = (in_len + num_of_padding) / t_len
    else:
        n_token = math.ceil((in_len - (t_len - step)) / step)
    return int(n_token)


def plot_data_set(dataset_base_path: str,
                  organize_type: tuple,
                  dataset_type: tuple,
                  is_show_inv_std: bool = False,
                  is_write_img: bool = True,
                  image_write_path: str = None) -> None:
    """"""
    plt.ioff()
    assert os.path.exists(dataset_base_path)
    if is_write_img:
        assert os.path.exists(image_write_path)

    data_time_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for og_type in organize_type:
        for dt_type in dataset_type:
            curr_dataset_type = og_type + '_' + dt_type
            temp_set_fold_path = os.path.join(dataset_base_path, og_type, 'data_set', dt_type)
            temp_set_list = os.listdir(temp_set_fold_path)

            x = [x for x in range(161)]
            if is_show_inv_std and dt_type == 'standardized':
                temp_std_path = os.path.join(dataset_base_path, og_type, 'data_set', 'standardizer.pickle')
                temp_stder = file_operation.load_dic_in_pickle(temp_std_path)['standardizer']['standardizer']
                fig, ax = plt.subplots(3, 2)
            else:
                fig, ax = plt.subplots(3)
            fig.suptitle(f'{curr_dataset_type}')

            for i, s in enumerate(temp_set_list):
                temp_file_fold_path = os.path.join(temp_set_fold_path, s)
                temp_file_list = os.listdir(temp_file_fold_path)

                if is_show_inv_std and dt_type == 'standardized':
                    ax[i][0].set_title(f'{s}_std_data')
                    ax[i][1].set_title(f'{s}_inv_std_data')
                else:
                    ax[i].set_title(f'{s}')

                for file in temp_file_list:
                    temp_file_path = os.path.join(temp_file_fold_path, file)
                    temp_file = np.load(temp_file_path)
                    temp_data = temp_file[0:-1]
                    # temp_data_label = temp_file[-1]

                    if is_show_inv_std and dt_type == 'standardized':
                        temp_data_cp = np.copy(temp_data)
                        temp_inv_data = temp_stder.inverse_transform(temp_data_cp)
                        ax[i][0].plot(x, temp_data)
                        ax[i][1].plot(x, temp_inv_data)
                    else:
                        ax[i].plot(x, temp_data)

            if is_write_img:
                # save fig
                save_name = curr_dataset_type + '_' + data_time_str + '.png'
                save_path = os.path.join(image_write_path, save_name)
                fig.savefig(save_path)


def merge_dict(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def cut_data_with_label(data_arr: np.array, out_len: int) -> np.array:
    """
     input_shape -- (n_sample, dim)
    """
    assert len(data_arr.shape) == 2
    assert data_arr.shape[-1] >= out_len

    cut_arr = data_arr[:, 0:out_len]
    return cut_arr


def tokenize_arr(data_arr: np.array, t_len: int, is_overlap: bool, step: int) -> np.array:
    """
    input_shape -- (n_sample, dim)
    output_shape -- (n_sample, n_token, t_len)
    """
    assert len(data_arr.shape) == 2
    arr_size = data_arr.shape
    row_idx = [x for x in range(arr_size[0])]
    token_arr_list = []
    for r_idx in row_idx:
        temp_token_arr = tokenize(data_arr[r_idx, :], t_len, is_overlap, step)
        temp_token_arr = np.expand_dims(temp_token_arr, axis=0)
        token_arr_list.append(temp_token_arr)
    token_arr = np.concatenate(token_arr_list, axis=0)
    return token_arr


def classification_preprocess(data_dict: dict,
                              transformation: tuple,
                              transformation_para: tuple) -> dict:
    """
    input_shape -- (n_sample, |data | label|)
    """
    assert data_dict is not {}
    # get data
    data_set_dict = {
        'train df': data_dict['train df'],
        'val df': data_dict['val df'],
        'test df': data_dict['test df'],
    }

    trans_data_dict = {}
    # loop data dict
    for key in data_set_dict:
        temp_df = data_set_dict[key]
        temp_data = temp_df.iloc[:, 0:-1].to_numpy(dtype='float32')
        temp_label = temp_df.iloc[:, -1].to_numpy(dtype='int64')
        # zip transform
        zip_trans = zip(transformation, transformation_para)
        for trans, trans_para in zip_trans:
            temp_data = trans(temp_data, **trans_para)
        temp_trans_data = temp_data
        temp_trans_data_set_dict = {
            'data': temp_trans_data,
            'label': temp_label
        }
        trans_data_dict.update({key: temp_trans_data_set_dict})
    return trans_data_dict


def miss_rate(labels: np.ndarray, outs: np.ndarray):
    """"""
    assert labels.shape == outs.shape
    n_samples = labels.shape[0]
    n_neg = np.where(labels == 0)[0].shape[0]
    counts = 0
    for i in range(n_samples):
        temp_label = labels[i]
        temp_out = outs[i]
        if (temp_label == 0) and (temp_label != temp_out):
            counts += 1
    mr_neg = counts / n_neg
    mr_total = counts / n_samples
    return mr_neg, mr_total


if __name__ == '__main__':
    data_set_base_path = 'D:\\workspace\\PycharmProjects\\battery_dataset'
    plot_out_path = os.path.join(data_set_base_path, 'data_set_image')
    file_organized_type_tup = ('2600P-01_DataSet_Balance', '2600P-01_DataSet_UnBalance')
    data_set_type_tup = ('none', 'standardized')
    plot_data_set(data_set_base_path,
                  file_organized_type_tup,
                  data_set_type_tup,
                  is_show_inv_std=True,
                  is_write_img=True,
                  image_write_path=plot_out_path)
    sys.exit(0)

