#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang
# Time: 2021/7/19
""""""
import os
import shutil
import pandas as pd
import pickle
import sys
import openpyxl


def path_assert(path: str, is_creat: bool = False):
    """"""
    if is_creat:
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        assert os.path.exists(path)


def DeleteFile(strFileName):
    """ 删除文件 """
    fileName = str(strFileName)
    if os.path.isfile(fileName):
        try:
            os.remove(fileName)
        except:
            pass


def Delete_File_Dir(dirName, flag = True):
    """ 删除指定目录，首先删除指定目录下的文件和子文件夹，然后再删除该文件夹 """
    if flag:
        dirName = str(dirName)
        """ 如何是文件直接删除 """
    if os.path.isfile(dirName):
        try:
            os.remove(dirName)
        except:
            pass
    elif os.path.isdir(dirName):
        """ 如果是文件夹，则首先删除文件夹下文件和子文件夹，再删除文件夹 """
        for item in os.listdir(dirName):
            tf = os.path.join(dirName,item)
            Delete_File_Dir(tf, False)
            """ 递归调用 """
        try:
            os.rmdir(dirName)
        except:
            pass


def move_file_to_fold(src, dst, file_list):
    if os.path.exists(src) and os.path.exists(dst):
        list_iter = iter(file_list)
        for file in list_iter:
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            shutil.move(src_file, dst_file)


def read_txt(file_path):
    assert os.path.exists(file_path)
    data = None
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data


def write_txt(file_path, data):
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            print(data, file=file)
    else:
        with open(file_path, 'a', encoding='utf-8') as file:
            print(data, file=file)
        

def xlsx_to_csv(src_path, out_path, encoding='utf-8', is_ver=False, ver_file_save_path=None):
    """将目标文件夹下的xlsx文件转换为csv文件并存储到输出文件夹"""
    if is_ver:
        assert os.path.exists(ver_file_save_path)
    xlsx_list = os.listdir(src_path)
    for i in iter(xlsx_list):
        file_path = os.path.join(src_path, i)
        xlsx_dic = pd.read_excel(file_path, sheet_name=None, header=0, index_col=0, dtype=str)
        for key in xlsx_dic.keys():
            temp_sheet_df_save_name = i + '_' + key
            temp_sheet_df = xlsx_dic[key]
            target_path = os.path.join(out_path, temp_sheet_df_save_name + '.csv')
            temp_sheet_df.to_csv(target_path, encoding=encoding)

            if is_ver:
                csv_df = pd.read_csv(target_path, header=0, index_col=0, dtype=str, low_memory=False)
                comp_res = csv_df.compare(temp_sheet_df)
                comp_file_save_name = os.path.join(ver_file_save_path, f'{temp_sheet_df_save_name}_comp_ver.csv')
                comp_res.to_csv(comp_file_save_name, encoding='utf8')


def save_dic_as_pickle(target_path, data_dic):
    """保存为pickle文件"""
    if not os.path.exists(target_path):
        with open(target_path, 'wb') as f:
            pickle.dump(data_dic, f)
    else:
        print(f'Path Exist: {target_path}')
        sys.exit(0)


def load_dic_in_pickle(source_path):
    """读取pickle文件"""
    if os.path.exists(source_path):
        file_name = os.path.basename(source_path)
        file_name = os.path.splitext(file_name)[0]
        dic = {f'{file_name}': {}}
        with open(source_path, 'rb') as f:
            temp_dic = pickle.load(f)
            dic[f'{file_name}'].update(temp_dic)
            return dic
    else:
        print(f'Path Not Exist: {source_path}')
        sys.exit(0)


def write_to_xlsx(target_path, data_dic):
    """"""
    if not os.path.exists(target_path):
        with pd.ExcelWriter(target_path, engine="xlsxwriter") as writer:
            for name in iter(data_dic.keys()):
                data_dic[name].to_excel(writer, sheet_name=name)
    else:
        print(f'Path Exist: {target_path}')
        sys.exit(0)


def read_xlsx_all_sheet(source_path: str) -> dict:
    """"""
    assert os.path.exists(source_path)
    file_name = os.path.basename(source_path)
    file_name = os.path.splitext(file_name)[0]
    wb = openpyxl.load_workbook(source_path)
    sheets = wb.sheetnames
    dic = {f'{file_name}': {}}
    for sheet in sheets:
        temp_sheet_df = pd.read_excel(source_path, sheet_name=sheet)
        temp_dic = {f'{sheet}': temp_sheet_df}
        dic[f'{file_name}'].update(temp_dic)
    return dic


if __name__ == '__main__':

    base = 'D:\\workspace\\PycharmProjects\\battery_dataset'
    src = os.path.join(base, 'raw_data', '2600P-01', 'static')
    dst = os.path.join(base, 'raw_data', '2600P-01', 'static_csv')

    xlsx_to_csv(src, dst, is_ver=True, ver_file_save_path=base + '\\data_compared')
    sys.exit(0)
