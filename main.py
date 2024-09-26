# -*- coding: UTF-8 -*-
"""
@Time     : 2022/3/23 11:02
@Author   : Caiming Liu
@Version  : V1
@File     : main.py
@Software : PyCharm
"""

# Local Modules
import sys
import os
import ctypes
import win32con

# Third-party Modules
import torch
import threading
from PySide2.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QVBoxLayout, QWidget, QTableWidgetItem
from PySide2.QtUiTools import QUiLoader
from PySide2.QtGui import QFont, QPixmap, QImage, QTextCursor
from PySide2.QtCore import Qt, QObject, QThread, Signal, SIGNAL
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn import preprocessing
from datetime import datetime
from utils.logger import setlogger
import logging
import argparse

from message_signal import MyMessageSignal
import numpy as np
import joblib  # 这个库在安装sklearn时好像会一起安装
from pylab import*

# Self-written Modules
from training_model import training_with_1D_CNN, training_with_LSTM, training_with_GRU, training_with_random_forest, training_with_svm
import datasets
from preprocess_train_result import plot_history_curcvs, plot_confusion_matrix, brief_classification_report, plot_metrics
from preprocess_train_result import dl_plot_history_curcvs, dl_plot_confusion_matrix, dl_brief_classification_report, dl_plot_metrics
from preprocess_train_result import dtl_plot_history_curcvs, dtl_plot_confusion_matrix, dtl_brief_classification_report, dtl_plot_metrics
from diagnosis import diagnosis
from utils.train_diagnosis_utils import train_diagnosis_utils
from utils.train_diagnosis_DTL_utils import train_diagnosis_DTL_utils
from preprocess import load_data
from model_train import model_training
import global_var as gl
from cls_model_define import *
import quality_trace_src
import pandas as pd

args = None
# 定义一些全局变量
global training_end_signal  # 定义一个信号，用于当模型训练完成之后通知主线程进行弹窗提示
training_end_signal = MyMessageSignal()

global diagnosis_end_signal  # 诊断结束信号
diagnosis_end_signal = MyMessageSignal()


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

cwru_data_list = {'97': '正常基座数据',
                  '105': '内圈故障（故障直径为0.1778mm）', '118': '滚动体故障（故障直径为0.1778mm）', '130': '6点方向外圈故障（故障直径为0.1778mm）',
                  '169': '内圈故障（故障直径为0.3556mm）', '185': '滚动体故障（故障直径为0.3556mm）', '197': '6点方向外圈故障（故障直径为0.3556mm）',
                  '209': '内圈故障（故障直径为0.5334mm）', '222': '滚动体故障（故障直径为0.5334mm）', '234': '6点方向外圈故障（故障直径为0.5334mm）'}

class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)

class EmittingStream(QObject):
    textWritten = Signal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):  # real signature unknown; restored from __doc__
        """ flush(self) """
        pass



class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.init_UI()
        # 定义并初始化一些成员变量
        self.data_file_path = ''  # 初始化一个 数据所在的 文件路径
        self.model_file_path = ''  # 初始化一个 模型所在的 文件路径
        self.cache_path = os.getcwd() + '/cache'  # 所有图片等的缓存路径
        self.training_flag = False  # 是否有模型在训练的标志位
        self.model_name = ''  # 初始化一个模型名字
        self.model = ''  # 训练得到的模型
        self.classification_report = ''  # 初始化一个 分类报告
        self.score = ''  # 初始化一个模型得分

    def init_UI(self):
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load('UI/comb_main.ui')

        self.ui.pb_mt_select_file.clicked.connect(self.mt_select_file)
        self.ui.pb_mt_visual_data.clicked.connect(self.mt_visual_data)
        self.ui.pb_mt_start_training.clicked.connect(self.mt_start_training)
        self.ui.pb_mt_show_result.clicked.connect(self.mt_show_result)
        self.ui.pb_mt_save_model.clicked.connect(self.mt_save_model)
        self.ui.pb_fd_select_model.clicked.connect(self.fd_select_model)
        self.ui.pb_fd_real_time_diagnosis.clicked.connect(self.fd_real_time_diagnosis)
        self.ui.pb_fd_local_diagnosis.clicked.connect(self.fd_local_diagnosis)
        #self.ui.actionabout.clicked.connect(self.action_about)

        self.mt_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.mt_mlp_layout = QVBoxLayout(self.ui.l_mt_visual_data)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.mt_toolbar = NavigationToolbar(self.mt_canvas, self)
        self.mt_mlp_layout.addWidget(self.mt_toolbar)
        self.mt_mlp_layout.addWidget(self.mt_canvas)

        self.fd_canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.fd_mlp_layout = QVBoxLayout(self.ui.l_fd_visual_diagnosis_data)
        # Create toolbar, passing canvas as first parament, parent (self, the MainWindow) as second.
        self.fd_toolbar = NavigationToolbar(self.fd_canvas, self)
        self.fd_mlp_layout.addWidget(self.fd_toolbar)
        self.fd_mlp_layout.addWidget(self.fd_canvas)

        self.ui.pb_pa_show_result.clicked.connect(self.pa_show_result)

        sys.stdout = EmittingStream()
        self.connect(sys.stdout, SIGNAL('textWritten(QString)'), self.normalOutputWritten)
        sys.stder = EmittingStream()
        self.connect(sys.stder, SIGNAL('textWritten(QString)'), self.normalOutputWritten)

        self.ui.pb_pa_data_process.clicked.connect(self.pa_data_process)
        self.ui.pb_pa_start_training.clicked.connect(self.pa_start_training)
        self.ui.pb_pa_end_training.clicked.connect(self.pa_end_training)


        self.ui.pb_qt_start_quality_trace.clicked.connect(self.qt_start_quality_trace)

    # 质量追溯代码
    def qt_start_quality_trace(self):
        self.ui.pb_qt_start_quality_trace.setEnabled(False)
        # load dataset file
        data_file_path = '.\\quality_trace_data\\data_set.pt'

        # dataset_dict
        # |    key     |      value type     |   size (n_samples, features)       | note            |
        # | 'train df' |    pandas.DataFrame | (28160, 162)                       | train data      |
        # | 'val df'   |    pandas.DataFrame | (9384, 162)                        | validation data |
        # | 'test df'  |    pandas.DataFrame | (9384, 162)                        | test data       |
        dataset_dict = quality_trace_src.file_operation.load_dic_in_pickle(data_file_path)['data_set']

        # load model
        model_path = '.\\quality_trace_model\\MyModel_290.pt'
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model = model_dict['model']

        # data pre-process
        input_len = model_dict['in_len']
        tokenize_para = {
            't_len': model_dict['t_len'],
            'is_overlap': False,
            'step': 16,
        }
        n_token = quality_trace_src.functional.cal_n_token(input_len, **tokenize_para)
        transformation = (quality_trace_src.functional.cut_data_with_label,
                          quality_trace_src.functional.tokenize_arr,
                          )
        transformation_para = (
            {'out_len': input_len},  # (out_len, )
            tokenize_para,  # (t_len, is_overlap, step)
        )
        preprocess_data_dict = quality_trace_src.functional.classification_preprocess(dataset_dict,
                                                                        transformation,
                                                                        transformation_para)

        # creat dataloader for test
        test_dataset = preprocess_data_dict['test df']
        test_data_set = quality_trace_src.data_set_load.ClsDataSetCreator.creat_dataset(test_dataset,
                                                                          bsz=2,
                                                                          is_shuffle=True,
                                                                          num_of_worker=0)
        # set model to eval mode
        model.eval()
        result_list = []
        label_list = []
        with torch.no_grad():
            for batch_idx, zip_data in enumerate(test_data_set):
                # data :(Batch_size, number of token, token len)
                # label: (number of class)
                data, label = zip_data

                # out:(Batch_size, number of class)
                # if a defective cell, then [1, 0]
                # if a normal cell, then [0, 1]
                out = model(data)
                out.unsqueeze_(1)
                label.unsqueeze_(1)
                result_list.append(out)
                label_list.append(label)
                if batch_idx > 160:
                    break
        battery_num = 50
        result_list = torch.cat(result_list, dim=0)
        result_list = result_list.cpu().numpy()
        result_list = np.squeeze(result_list, 1)
        table_list = result_list[:battery_num, 0]
        result_list = result_list[:battery_num, 0]

        label_list = torch.cat(label_list, dim=0)
        label_list = label_list.cpu().numpy()
        label_list = np.squeeze(label_list, 1)
        label_list = label_list[:battery_num]

        mat = result_list.reshape(10, int(battery_num//10))
        plt.matshow(mat, cmap=plt.cm.Blues)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(x=j, y=i, s=i * 5 + j)
        plt.savefig(self.cache_path + '/quality_trace.png', dpi=150, bbox_inches='tight')
        #plt.show()
        img = QImage(self.cache_path + '/quality_trace.png')
        img_result = img.scaled(self.ui.l_qt_visual_trace_result.width(), self.ui.l_qt_visual_trace_result.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.l_qt_visual_trace_result.setPixmap(QPixmap.fromImage(img_result))


        for row in range(battery_num):
            item = QTableWidgetItem()  # 模型
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 文本显示位置
            item.setText(f"{row + 1}")
            item.setFont(QFont("微软雅黑", 12, QFont.Black))  # 设置字体
            self.ui.tw_qt_result.setItem(row, 0, item)

        for row in range(battery_num):
            item = QTableWidgetItem()  # 模型
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 文本显示位置
            item_text = ''
            if int(round(table_list[row])) == 0:
                item_text = "正常"
            else:
                item_text = "异常"
            item.setText(item_text)
            item.setFont(QFont("微软雅黑", 12, QFont.Black))  # 设置字体
            self.ui.tw_qt_result.setItem(row, 1, item)

        for row in range(battery_num):
            item = QTableWidgetItem()  # 模型
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter)  # 文本显示位置
            item_text = ''
            if int(round(label_list[row])) == 1:
                item_text = "正常"
            else:
                item_text = "异常"
            item.setText(item_text)
            item.setFont(QFont("微软雅黑", 12, QFont.Black))  # 设置字体
            self.ui.tw_qt_result.setItem(row, 2, item)


        self.ui.pb_qt_start_quality_trace.setEnabled(True)

    # 工艺分析代码
    def pa_show_result(self):

        show_mode = self.ui.buttonGroup_pa.checkedId()
        # print(show_mode)
        # TODO: 这里的 Id 自己测出来的, 应该还有别的方法直接得到所选框的内容
        self.hyper_para()
        show_result = self.ui.comb_pa_abnoramal_report.currentText()
        # show_result = str(show_result)

        if -2 == show_mode:  # 展示结果
            # 读取图片文件，进行显示
            img1 = QImage(f'plots/{gl.model}_{gl.dataset}/{show_result}.png')
            img_result1 = img1.scaled(self.ui.l_pa_train_result.width(), self.ui.l_pa_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_pa_train_result.setPixmap(QPixmap.fromImage(img_result1))

        elif -3 == show_mode:  # 展示ROC曲线
            # 读取图片文件，进行显示
            img = QImage(f'plots/{gl.model}_{gl.dataset}/{gl.model}_{gl.dataset}-ROC-Curves.png')
            img_result = img.scaled(self.ui.l_pa_visual_data.width(), self.ui.l_pa_visual_data.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_pa_visual_data.setPixmap(QPixmap.fromImage(img_result))
        elif -4 == show_mode:  # 展示精度召回曲线
            # 读取图片文件，进行显示
            img = QImage(f'plots/{gl.model}_{gl.dataset}/{gl.model}_{gl.dataset}-PlotPrecisionRecallCurve.png')
            img_result = img.scaled(self.ui.l_pa_visual_data.width(), self.ui.l_pa_visual_data.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_pa_visual_data.setPixmap(QPixmap.fromImage(img_result))
        elif -5 == show_mode:  # 展示 损失曲线
            # 读取图片文件，进行显示
            img = QImage(f'plots/{gl.model}_{gl.dataset}/{gl.model}_{gl.dataset}-training-graph.png')
            img_result = img.scaled(self.ui.l_pa_visual_data.width(), self.ui.l_pa_visual_data.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_pa_visual_data.setPixmap(QPixmap.fromImage(img_result))


    # 数据预处理
    def pa_data_process(self):
        self.ui.comb_pa_select_data.setEnabled(False)
        self.ui.comb_pa_select_data.setEnabled(True)
        data_set = self.ui.comb_pa_select_data.currentText()
        QMessageBox.information(self, "提示", "您选择的数据集为: " + data_set)
        load_data(str(data_set))
        QMessageBox.information(self, "提示", "数据预处理结束 !         ")

    #模型加载
    def pa_end_training(self):
        self.ui.pb_pa_end_training.setEnabled(False)
        self.ui.tb_pa_train_result.clear()
        self.hyper_para()
        # self.my_thread1.start()  # 启动线程
        self.ui.pb_pa_end_training.setEnabled(True)
        QMessageBox.information(self, "提示", "加载模型！   ")
        training_thread = threading.Thread(target=model_training,
                                           args=(self.ui.tw_pa_test_result, gl.model, gl.dataset, gl.Epoch, '--test')
                                           )
        training_thread.start()

    def hyper_para(self):
        model = self.ui.comb_pa_select_model.currentText()
        gl.model = str(model)
        print('模型为 %s' % gl.model)

        dataset = self.ui.comb_pa_select_data.currentText()
        gl.dataset = str(dataset)
        print('数据集为 %s' % gl.dataset)

        Epoch = self.ui.comb_pa_select_epoch.currentText()
        gl.Epoch = int(Epoch)
        print('迭代次数为 %d' % gl.Epoch)


    def pa_start_training(self):
        self.ui.pb_pa_start_training.setEnabled(False)
        self.ui.tb_pa_train_result.clear()
        self.hyper_para()
        self.ui.pb_pa_start_training.setEnabled(True)
        QMessageBox.information(self, "提示", "开始训练！   ")
        training_thread = threading.Thread(target=model_training,
                                           args=(self.ui.tw_pa_test_result,gl.model, gl.dataset, gl.Epoch, '--retrain')
                                           )
        training_thread.start()


    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.ui.tb_pa_train_result.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(text)
        self.ui.tb_pa_train_result.setTextCursor(cursor)
        self.ui.tb_pa_train_result.ensureCursorVisible()

    # 故障诊断部分代码
    def action_about(self):
        about_text =  "Auther : liucaiming\n"
        about_text += "Data   : 2022/04/12\n"
        about_text += "Version: V0\n"
        about_text += "(C) Notepad (R)\n"
        QMessageBox.about(self, "About this", about_text)

    def mt_select_file(self):
        self.ui.pb_mt_select_file.setEnabled(False)
        '''file_path, _ = QFileDialog.getOpenFileName(self,
                                                   '打开文件',  # 标题
                                                   './data/CWRU/12k Drive End Bearing Fault Data',  # 默认开始打开的文件路径， . 表示在当前文件夹下
                                                    '(*.mat)'  # 文件过滤，表示只显示后缀名为 .mat 的文件
                                                   )'''
        file_path, _ = QFileDialog.getOpenFileName(self,
                                                           '打开文件',  # 标题
                                                           './data',  # 默认开始打开的文件路径， . 表示在当前文件夹下
                                                           )
        if '' != file_path:  # 选择了文件, 则将路径更新，否则，保留原路径
            self.data_file_path = file_path
            self.ui.tb_mt_train_result.setText('选择文件：' + self.data_file_path + '\n--------------')
        self.ui.pb_mt_select_file.setEnabled(True)

    def mt_visual_data(self):
        self.ui.pb_mt_visual_data.setEnabled(False)
        self.mt_canvas.axes.cla()  # Clear the canvas.
        if '' == self.data_file_path:  # 没有选择过文件
            reply = QMessageBox.information(self, '提示', '请先选择文件！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.ui.pb_mt_visual_data.setEnabled(True)
                return  # 直接退出
        '''file = loadmat(self.data_file_path)  # 加载文件，这里得到的文件是一个字典
        file_keys = file.keys()
        for key in file_keys:
            if 'DE' in key:  # DE: 驱动端测得的振动数据
                global data  # 定义一个全局变量
                data = file[key][:2048]  # 截取数据的前2500个数据点进行绘图'''
        file = pd.read_csv(self.data_file_path, encoding='gbk', usecols=[0])
        data = file.values[:128].reshape(-1, 1)
        self.mt_canvas.axes.plot(data)
        self.mt_canvas.draw()
        self.ui.pb_mt_visual_data.setEnabled(True)

    def mt_start_training(self):
        if self.training_flag:  # 有模型在训练
            reply = QMessageBox.information(self, '提示', '正在训练模型，请等待...', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return  # 退出函数

        if '' == self.data_file_path:  # 没有选择过文件
            reply = QMessageBox.information(self, '提示', '请先选择文件！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return  # 退出函数

        # 到这里，就是 没有模型在训练，且选择了文件
        # 提示用户确认
        select_model = self.ui.comb_mt_select_model.currentText()  # 用户选择的 模型
        reply = QMessageBox.information(self, '提示', '确定使用“' + select_model + '”进行训练。\n请确保所有数据在一个文件夹下！',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.No:
            return  # 退出函数

        # 到这里，可以开始训练了
        self.training_flag = True  # 改变标志位
        self.ui.statusbar.showMessage('正在训练模型...', 120)

        # 不同的模型，参数设置不一样
        if '随机森林' == select_model or 'SVM' == select_model:
            signal_length = 128
            signal_number = 1000  # 每个类别中要抽取的样本的数量
            normal = False  # 是否标准化
        else:
            signal_length = 128
            signal_number = 1000  # 每个类别中要抽取的样本的数量
            normal = True  # 是否标准化
        # 获得数据所在的文件夹  E:/fff/hhh/iii/tt.mat --> tt.mat --> E:/fff/hhh/iii/ --> E:/fff/hhh/iii --> E:/fff/hhh
        data_path = self.data_file_path.split('/')[-1]  # 先获得文件名
        data_path = self.data_file_path.split(data_path)[0]  # 再去掉文件路径中的文件名
        data_path = data_path.rsplit("/", 1)[0]
        data_path = data_path.rsplit("/", 1)[0]

        if 'RF' == select_model:
            self.model_name = 'random_forest'
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            self.ui.tb_mt_train_result.setText(text + '\n模型选择：随机森林\n正在训练模型...\n--------------')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=random_forest_training,
                                               args=(data_path, signal_length, signal_number, normal,
                                                     self.cache_path, self.model_name)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
            training_end_signal.send_msg.connect(self.training_end_slot)  # 信号与槽连接

        elif 'SVM' == select_model:
            self.model_name = 'SVM'
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            self.ui.tb_mt_train_result.setText(text + '\n模型选择：SVM\n正在训练模型...\n--------------')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=svm_training,
                                               args=(data_path, signal_length, signal_number, normal,
                                                     self.cache_path, self.model_name)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
            training_end_signal.send_msg.connect(self.training_end_slot)  # 信号与槽连接
        elif 'DAN' == select_model:
            self.model_name = select_model
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            if torch.cuda.is_available() and False:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n检测到GPU可用\n--------------')
            else:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n未检测到可用GPU\n--------------')
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            self.ui.tb_mt_train_result.setText(text + '\n正在训练模型...\n--------------')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=SDTL_model_training,
                                               args=(data_path, signal_length, signal_number, normal,
                                                     self.cache_path, self.model_name)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
            training_end_signal.send_msg.connect(self.training_end_slot)  # 信号与槽连接
        elif 'MFSAN' == select_model:
            self.model_name = select_model
            self.model_name='MINE'#暂时这里选择
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            if torch.cuda.is_available() and False:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n检测到GPU可用\n--------------')
            else:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n未检测到可用GPU\n--------------')
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            self.ui.tb_mt_train_result.setText(text + '\n正在训练模型...\n--------------')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=MDTL_model_training,
                                               args=(data_path, signal_length, signal_number, normal,
                                                     self.cache_path, self.model_name)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
            training_end_signal.send_msg.connect(self.training_end_slot)  # 信号与槽连接
        elif 'DG' == select_model:
            self.model_name = select_model
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            if torch.cuda.is_available() and False:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n检测到GPU可用\n--------------')
            else:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n未检测到可用GPU\n--------------')
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            self.ui.tb_mt_train_result.setText(text + '\n正在训练模型...\n--------------')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=MDTL_model_training,
                                               args=(data_path, signal_length, signal_number, normal,
                                                     self.cache_path, self.model_name)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
            training_end_signal.send_msg.connect(self.training_end_slot)  # 信号与槽连接
        else:
            self.model_name = select_model
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            if torch.cuda.is_available() and False:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n检测到GPU可用\n--------------')
            else:
                self.ui.tb_mt_train_result.setText(text + '\n模型选择：' + self.model_name + '\n未检测到可用GPU\n--------------')
            text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
            self.ui.tb_mt_train_result.setText(text + '\n正在训练模型...\n--------------')

            # 创建子线程，训练模型
            training_thread = threading.Thread(target=DL_model_training,
                                               args=(data_path, signal_length, signal_number, normal,
                                                     self.cache_path, self.model_name)
                                               )
            # training_thread.setDaemon(True)  # 守护线程
            training_thread.start()
            training_end_signal.send_msg.connect(self.training_end_slot)  # 信号与槽连接



    def training_end_slot(self, msg):
        self.model = msg['model']
        self.classification_report = msg['classification_report']
        self.score = msg['score']

        QMessageBox.information(self, '提示', '训练完成！', QMessageBox.Yes, QMessageBox.Yes)
        self.ui.statusbar.close()
        text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
        self.ui.tb_mt_train_result.setText(text + '\n训练完成，模型得分：' + self.score + '\n--------------')
        self.ui.l_mt_train_result.setText(self.classification_report)
        self.training_flag = False

    def diagnosis_end_slot(self, msg):
        pred_result = msg['pred_result']
        text = self.ui.tb_fd_diagnosis_result.toPlainText()
        self.ui.tb_fd_diagnosis_result.setText(text + '\n诊断结果：' + pred_result + '\n--------------')
        self.ui.pb_fd_real_time_diagnosis.setEnabled(True)
        self.ui.pb_fd_local_diagnosis.setEnabled(True)

    def mt_show_result(self):
        if '' == self.model_name:  # 说明还没有训练过模型
            reply = QMessageBox.information(self, '提示', '你还没有训练模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return

        show_mode = self.ui.buttonGroup_fd.checkedId()
        # print(show_mode)
        # TODO: 这里的 Id 自己测出来的, 应该还有别的方法直接得到所选框的内容
        if -2 == show_mode:  # 展示 分类报告
            self.ui.l_mt_train_result.setText(self.classification_report)
        elif -3 == show_mode:  # 展示 混淆矩阵
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_confusion_matrix.png')
            img_result = img.scaled(self.ui.l_mt_train_result.width(), self.ui.l_mt_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_mt_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -4 == show_mode:  # 展示 ROC曲线
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_ROC_Curves.png')
            img_result = img.scaled(self.ui.l_mt_train_result.width(), self.ui.l_mt_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_mt_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -5 == show_mode:  # 展示 精度召回曲线
            # 读取图片文件，进行显示
            img = QImage(self.cache_path + '/' + self.model_name + '_Precision_Recall_Curves.png')
            img_result = img.scaled(self.ui.l_mt_train_result.width(), self.ui.l_mt_train_result.height(),  # 裁剪图片将图片大小
                                    Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            self.ui.l_mt_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -6 == show_mode:  # 展示 损失曲线
            if 'random_forest' == self.model_name:  # 随机森林没有损失曲线
                QMessageBox.information(self, '提示', '随机森林模型没有损失曲线哦！', QMessageBox.Yes, QMessageBox.Yes)
            elif 'SVM' == self.model_name:
                QMessageBox.information(self, '提示', 'SVM模型没有损失曲线哦！', QMessageBox.Yes, QMessageBox.Yes)
            else:
                # 读取图片文件，进行显示
                img = QImage(self.cache_path + '/' + self.model_name + '_train_valid_loss.png')
                img_result = img.scaled(self.ui.l_mt_train_result.width(), self.ui.l_mt_train_result.height(),
                                        # 裁剪图片将图片大小
                                        Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.ui.l_mt_train_result.setPixmap(QPixmap.fromImage(img_result))
        elif -7 == show_mode:  # 展示 正确率曲线
            if 'random_forest' == self.model_name:  # 随机森林和SVM没有正确率曲线
                QMessageBox.information(self, '提示', '随机森林模型没有正确率曲线哦！', QMessageBox.Yes, QMessageBox.Yes)
            elif 'SVM' == self.model_name:
                QMessageBox.information(self, '提示', 'SVM模型没有正确率曲线哦！', QMessageBox.Yes, QMessageBox.Yes)
            else:
                # 读取图片文件，进行显示
                img = QImage(self.cache_path + '/' + self.model_name + '_train_valid_acc.png')
                img_result = img.scaled(self.ui.l_mt_train_result.width(), self.ui.l_mt_train_result.height(),
                                        # 裁剪图片将图片大小
                                        Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
                self.ui.l_mt_train_result.setPixmap(QPixmap.fromImage(img_result))

    def mt_save_model(self):
        if '' == self.model_name:  # 说明还没有训练过模型
            reply = QMessageBox.information(self, '提示', '你还没有训练模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return

        if 'random_forest' == self.model_name or 'SVM' == self.model_name:
            save_path, _ = QFileDialog.getSaveFileName(self,
                                                       '保存文件',  # 标题
                                                       './saved_models/' + self.model_name + '.m',
                                                       # 默认开始打开的文件路径， . 表示在当前文件夹下, 后面接的默认文件名
                                                       '(*.m)'  # 文件过滤，表示只显示后缀名为 .m 的文件
                                                       )
            if '' == save_path:  # 没有确定保存。这里也可以通过 变量 _ 来判断
                return
            # print(save_path)
            joblib.dump(self.model, save_path)  # 存储
        else:
            save_path, _ = QFileDialog.getSaveFileName(self, '保存文件', './saved_models/' + self.model_name + '.pth', '(*.pth)')
            if '' == save_path:  # 没有确定保存。这里也可以通过 变量 _ 来判断
                return
            #self.model.save(save_path)
            torch.save(self.model, save_path)
        text = self.ui.tb_mt_train_result.toPlainText()  # 获得原本显示的文字
        self.ui.tb_mt_train_result.setText(text + "\n模型保存成功\n--------------")

    def fd_select_model(self):
        self.ui.pb_fd_select_model.setEnabled(False)
        file_path, _ = QFileDialog.getOpenFileName(self, '选择模型', './saved_models/', '(*.m *.h5 *.pth)')
        if '' != file_path:  # 选择了文件, 则将路径更新，否则，保留原路径
            self.model_file_path = file_path
            self.ui.tb_fd_diagnosis_result.setText('选择文件：' + self.model_file_path + '\n--------------')
        self.ui.pb_fd_select_model.setEnabled(True)

    def fd_visual_data(self, data_path):
        self.fd_canvas.axes.cla()  # Clear the canvas.
        if '' == data_path:  # 没有选择过文件
            reply = QMessageBox.information(self, '提示', '请先选择文件！', QMessageBox.Yes, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                return  # 直接退出
        '''file = loadmat(data_path)  # 加载文件，这里得到的文件是一个字典
        cwru_key = data_path.split('/')[-1][:-4]
        file_keys = file.keys()
        for key in file_keys:
            if 'DE' in key:  # DE: 驱动端测得的振动数据
                global data  # 定义一个全局变量
                data = file[key][:2048]  # 截取数据的前2500个数据点进行绘图'''
        file = pd.read_csv(data_path, encoding='gbk', usecols=[0])
        data = file.values[:128].reshape(-1, 1)
        self.fd_canvas.axes.plot(data)
        #self.fd_canvas.axes.set_title(cwru_data_list[cwru_key])
        self.fd_canvas.draw()

    def fd_real_time_diagnosis(self):
        self.ui.pb_fd_real_time_diagnosis.setEnabled(False)
        self.ui.pb_fd_local_diagnosis.setEnabled(False)  # 同一时间只能进行一种诊断

        if '' == self.model_file_path:  # 没有选择过模型
            reply = QMessageBox.information(self, '提示', '你还没有选择模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if QMessageBox.Yes == reply:
                self.ui.pb_fd_real_time_diagnosis.setEnabled(True)
                self.ui.pb_fd_local_diagnosis.setEnabled(True)
                return

        text = self.ui.tb_fd_diagnosis_result.toPlainText()
        self.ui.tb_fd_diagnosis_result.setText(text + '\n实时诊断：正在采集数据...\n--------------')

        # TODO: 这里通过读取指定的文件夹数据来模拟实时采集数据
        real_time_data_path = 'G:/dataset/CWRU/12k Drive End Bearing Fault Data/130.mat'
        # 读取完数据后，自动可视化数据
        visual_data_pic_path = self.fd_visual_data(real_time_data_path)
        # 读取图片文件，进行显示
        img = QImage(visual_data_pic_path)
        img_result = img.scaled(self.ui.l_fd_visual_diagnosis_data.width(), self.ui.l_fd_visual_diagnosis_data.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.l_fd_visual_diagnosis_data.setPixmap(QPixmap.fromImage(img_result))
        text = self.ui.tb_fd_diagnosis_result.toPlainText()
        self.ui.tb_fd_diagnosis_result.setText(text + '\n实时诊断：正在诊断..\n--------------')

        # 开个子线程进行故障诊断
        diagnosis_end_signal.send_msg.connect(self.diagnosis_end_slot)  # 信号与槽连接
        diagnosis_thread = threading.Thread(target=fault_diagnosis, args=(self.model_file_path, real_time_data_path))
        diagnosis_thread.start()

    def fd_local_diagnosis(self):
        self.ui.pb_fd_local_diagnosis.setEnabled(False)
        self.ui.pb_fd_real_time_diagnosis.setEnabled(False)  # 同一时间只能进行一种诊断

        file_path, _ = QFileDialog.getOpenFileName(self, '选择数据', './data')
        if '' == file_path:  # 没有选择文件，也就是退出了本地诊断
            self.ui.pb_fd_real_time_diagnosis.setEnabled(True)
            self.ui.pb_fd_local_diagnosis.setEnabled(True)
            return

        text = self.ui.tb_fd_diagnosis_result.toPlainText()
        self.ui.tb_fd_diagnosis_result.setText(text + '\n选择文件：' + file_path + '\n--------------')

        if '' == self.model_file_path:  # 没有选择过模型
            reply = QMessageBox.information(self, '提示', '你还没有选择模型哦！', QMessageBox.Yes, QMessageBox.Yes)
            if QMessageBox.Yes == reply:
                self.ui.pb_fd_real_time_diagnosis.setEnabled(True)
                self.ui.pb_fd_local_diagnosis.setEnabled(True)
                return

        text = self.ui.tb_fd_diagnosis_result.toPlainText()
        self.ui.tb_fd_diagnosis_result.setText(text + '\n本地诊断：正在读取数据...\n--------------')

        # 读取完数据后，自动可视化数据
        visual_data_pic_path = self.fd_visual_data(file_path)
        # 读取图片文件，进行显示
        img = QImage(visual_data_pic_path)
        img_result = img.scaled(self.ui.l_fd_visual_diagnosis_data.width(), self.ui.l_fd_visual_diagnosis_data.height(),  # 裁剪图片将图片大小
                                Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        self.ui.l_fd_visual_diagnosis_data.setPixmap(QPixmap.fromImage(img_result))

        text = self.ui.tb_fd_diagnosis_result.toPlainText()
        self.ui.tb_fd_diagnosis_result.setText(text + '\n本地诊断：正在诊断..\n--------------')

        # 开个子线程进行故障诊断
        diagnosis_end_signal.send_msg.connect(self.diagnosis_end_slot)  # 信号与槽连接
        diagnosis_thread = threading.Thread(target=fault_diagnosis,
                                            args=(self.model_file_path, file_path))
        diagnosis_thread.start()

    def closeEvent(self, event):
        '''
        重写关闭窗口函数：在点击关闭窗口后，将缓存文件夹下的文件全部删除
        :param event:
        :return:
        '''
        file_names = os.listdir(self.cache_path)
        for file_name in file_names:
            os.remove(self.cache_path + '/' + file_name)

        sys.exit()

def svm_training(data_path, signal_length, signal_number, normal, save_path, model_name):
    '''
        训练 随机森林 模型
        :param data_path: 数据路径
        :param signal_length: 信号长度
        :param signal_number: 信号个数
        :param normal: 是否标准化
        :param save_path: 训练完后各种图的保存路径
        :param model_name: 模型名字
        :return:
        '''
    if data_path.find('CWRU') != -1:
        data_name = 'CWRU'
    elif data_path.find('Paderborn') != -1:
        data_name = 'Paderborn'
    elif data_path.find('TractionSpeed') != -1:
        data_name = 'TractionSpeed'
    elif data_path.find('ScreedPressure') != -1:
        data_name = 'ScreedPressure'
    Dataset = getattr(datasets, data_name)  # 返回一个对象属性值
    source_train, source_test, _ = Dataset(data_path, [[0], [0]], 'mean-std', signal_length).data_split(transfer_learning=False)
    X_train = np.array(source_train.seq_data)
    X_train = np.squeeze(X_train, -1)
    y_train = np.array(source_train.labels)

    X_test = np.array(source_test.seq_data)
    X_test = np.squeeze(X_test, -1)
    y_test = np.array(source_test.labels)

    model, score, X_train_feature_extraction, X_test_feature_extraction = training_with_svm(X_train, y_train,
                                                                                                      X_test, y_test)
    plot_confusion_matrix(model, model_name, save_path, X_test_feature_extraction, y_test)  # 绘制混淆矩阵
    classification_report = brief_classification_report(model, model_name, X_test_feature_extraction, y_test)  # 计算分类报告
    plot_metrics(model, model_name, save_path, X_test_feature_extraction, y_test)  # 绘制 召回率曲线和精确度曲线
    # training_end_signal.run()
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score)}
    training_end_signal.send_msg.emit(msg)

def random_forest_training(data_path, signal_length, signal_number, normal, save_path, model_name):
    '''
    训练 随机森林 模型
    :param data_path: 数据路径
    :param signal_length: 信号长度
    :param signal_number: 信号个数
    :param normal: 是否标准化
    :param save_path: 训练完后各种图的保存路径
    :param model_name: 模型名字
    :return:
    '''
    data_name = 'CWRU'
    if data_path.find('CWRU') != -1:
        data_name = 'CWRU'
    elif data_path.find('Paderborn') != -1:
        data_name = 'Paderborn'
    elif data_path.find('TractionSpeed') != -1:
        data_name = 'TractionSpeed'
    elif data_path.find('ScreedPressure') != -1:
        data_name = 'ScreedPressure'
    Dataset = getattr(datasets, data_name)  # 返回一个对象属性值
    source_train, source_test, _ = Dataset(data_path, [[0], [0]], 'mean-std', signal_length).data_split(transfer_learning=False)
    #source_train, source_test = Dataset(data_path, 'mean-std', op=0).data_preprare(is_src=False)
    X_train = np.array(source_train.seq_data)
    X_train = np.squeeze(X_train, -1)
    y_train = np.array(source_train.labels)

    X_test = np.array(source_test.seq_data)
    X_test = np.squeeze(X_test, -1)
    y_test = np.array(source_test.labels)
    model, score, X_train_feature_extraction, X_test_feature_extraction = training_with_random_forest(X_train, y_train,
                                                                                                      X_test, y_test)
    # plot_history_curcvs(history, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线 --- 随机森林没有
    plot_confusion_matrix(model, model_name, save_path, X_test_feature_extraction, y_test)  # 绘制混淆矩阵
    classification_report = brief_classification_report(model, model_name, X_test_feature_extraction, y_test)  # 计算分类报告
    plot_metrics(model, model_name, save_path, X_test_feature_extraction, y_test)  # 绘制 召回率曲线和精确度曲线
    # sleep(3)

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户
    # training_end_signal.run()
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score)}
    training_end_signal.send_msg.emit(msg)


def DL_model_training(data_path, signal_length, signal_number, normal, save_path, model_name):
    args = parse_args_DL()
    args.model_name = model_name
    args.data_dir = data_path
    args.data_length = signal_length
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_diagnosis_utils(args, save_dir)
    #训练集，画图用
    val_dataloader = trainer.setup()
    model, loss_list, val_loss_list, acc_list, val_acc_list = trainer.train()

    model.eval()
    dl_plot_history_curcvs(loss_list, val_loss_list, acc_list, val_acc_list, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    dl_plot_confusion_matrix(model, model_name, save_path, val_dataloader)  # 绘制混淆矩阵
    classification_report = dl_brief_classification_report(model, model_name,val_dataloader)  # 计算分类报告
    dl_plot_metrics(model, model_name, save_path, val_dataloader)  # 绘制 召回率曲线和精确度曲线

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户, 同时将模型得分发送过去以便显示
    # training_end_signal.run()
    score = max(val_acc_list)
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score)}
    training_end_signal.send_msg.emit(msg)

def SDTL_model_training(data_path, signal_length, signal_number, normal, save_path, model_name):
    # 单源域适应
    args = parse_args_DTL()
    args.model_name = model_name
    args.data_dir = data_path
    args.data_length = signal_length
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_diagnosis_DTL_utils(args, save_dir)
    # 目标域验证集,画图用
    val_dataloader = trainer._init_data()
    model, loss_list, val_acc_list = trainer.train_1src_and_srccomb()
    model.eval()
    dtl_plot_history_curcvs(loss_list, val_acc_list, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    dtl_plot_confusion_matrix(model, model_name, save_path, val_dataloader)  # 绘制混淆矩阵
    classification_report = dtl_brief_classification_report(model, model_name,val_dataloader)  # 计算分类报告
    dtl_plot_metrics(model, model_name, save_path, val_dataloader)  # 绘制 召回率曲线和精确度曲线

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户, 同时将模型得分发送过去以便显示
    # training_end_signal.run()
    score = max(val_acc_list)
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score)}
    training_end_signal.send_msg.emit(msg)

def MDTL_model_training(data_path, signal_length, signal_number, normal, save_path, model_name):
    #多源域适应/泛化
    args = parse_args_DTL()
    args.model_name = model_name
    args.data_dir = data_path
    args.data_length = signal_length
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device.strip()
    # Prepare the saving path for the model
    sub_dir = args.model_name + '_' + datetime.datetime.strftime(datetime.datetime.now(), '%m%d-%H%M%S')
    save_dir = os.path.join(args.checkpoint_dir, sub_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # set the logger
    setlogger(os.path.join(save_dir, 'train.log'))

    # save the args
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))

    trainer = train_diagnosis_DTL_utils(args, save_dir)
    #目标域验证集,画图用
    val_dataloader = trainer._init_data()
    model, loss_list, val_acc_list = trainer.train_multi_src()
    model.eval()
    dtl_plot_history_curcvs(loss_list, val_acc_list, save_path, model_name)  # 绘制 训练集合验证集 损失曲线和正确率曲线
    dtl_plot_confusion_matrix(model, model_name, save_path, val_dataloader)  # 绘制混淆矩阵
    classification_report = dtl_brief_classification_report(model, model_name,val_dataloader)  # 计算分类报告
    dtl_plot_metrics(model, model_name, save_path, val_dataloader)  # 绘制 召回率曲线和精确度曲线

    # 发送信号通知主线程训练完成，让主线程发个弹窗，通知用户, 同时将模型得分发送过去以便显示
    # training_end_signal.run()
    score = max(val_acc_list)
    msg = {'model': model, 'classification_report': classification_report, 'score': str(score)}
    training_end_signal.send_msg.emit(msg)


def fault_diagnosis(model_file_path, real_time_data_path):
    '''
    使用模型进行故障诊断
    :param model_file_path: 模型路径
    :param real_time_data_path: 数据路径
    :return:
    '''
    suffix = model_file_path.split('/')[-1].split('.')[-1]  # 获得所选模型的后缀名
    if 'm' == suffix:  # 说明是随机森林
        diagnosis_samples = diagnosis_stage_prepro(real_time_data_path, 128, 1000, False)
        pred_result = diagnosis(diagnosis_samples, model_file_path)
    else:
        diagnosis_samples = diagnosis_stage_prepro(real_time_data_path, 128, 1000, True)
        pred_result = diagnosis(diagnosis_samples, model_file_path)

    # 诊断完成，将结果发送回去
    msg = {'pred_result': pred_result}
    diagnosis_end_signal.send_msg.emit(msg)

def diagnosis_stage_prepro(data_path, signal_length=128, signal_number=1000, normal=True):
    '''
    诊断阶段对数据的预处理
    :param data_path: 数据路径
    :param signal_length: 信号长度
    :param signal_number: 信号数量
    :param normal: 是否标准化
    :return:
    '''
    file_name = data_path.split('/')[-1].split('.')[0]  # 获得文件名

    def capture():
        """
        函数说明：读取mat文件，并将数据以字典返回（文件名作为key，数据作为value）

        Parameters:
            无
        Returns:
            data_dict : 数据字典
        """
        data_dict = {}

        '''file = loadmat(data_path)  # 读取 .mat 文件，返回的是一个 字典
        file_keys = file.keys()  # 获得该字典所有的key
        for key in file_keys:  # 遍历key, 获得 DE 的数据
            if 'DE' in key:  # DE: 驱动端 振动数据
                data_dict[file_name] = file[key].ravel()
        return data_dict'''

        file = pd.read_csv(data_path, encoding='gbk', usecols=[0])
        data_dict[file_name] = file.values.reshape(-1, 1)

        return data_dict

    def slice(data_dict):
        """
        函数说明：切取数据样本

        Parameters：
            data_dict : dict, 要进行划分的数据
        Returns:
            diagnosis_samples_dict : 切分后的 诊断样本
        """
        diagnosis_samples_dict = {}  # 训练集 样本

        keys = data_dict.keys()
        for key in keys:
            slice_data = data_dict[key]  # 获得value，即取得该文件里的 DE 数据
        all_lenght = len(slice_data)  # 获得数据长度
        sample_number = int(signal_number)  # 需要采集的信号 个数，防止输入小数，所以将其转为int

        samples = []  # 该文件中抽取的样本
        for j in range(sample_number):  # 在该文件中 抽取 信号，共抽取sample_number个（随机抽取）
            random_start = np.random.randint(low=0, high=(all_lenght - signal_length))  # high=(all_lenght - signal_length)：保证从任何一个位置开始都可以取到完整的数据长度
            sample = slice_data[random_start: random_start + signal_length]  # 抽取信号
            samples.append(sample)

        diagnosis_samples_dict[key] = samples  # 字典存储---文件名：对应的信号
        return diagnosis_samples_dict

    def scalar_stand(X_train):
        '''
        函数说明：用训练集标准差标准化训练集

        Parameters:
            X_train : 训练集
            X_valid_test : 验证和测试集
        Returns:
            X_train : 标准化后的训练集
        '''
        scalar = preprocessing.StandardScaler().fit(X_train)
        X_train = scalar.transform(X_train)
        return X_train

    # 从.mat文件中读取出数据的字典
    data_dict = capture()
    # 将数据按样本要求切分
    diagnosis_samples_dict = slice(data_dict)
    # diagnosis_samples = []
    diagnosis_samples = diagnosis_samples_dict[file_name]  # 取得对应文件中样本的数据
    # diagnosis_samples += x

    # 数据 是否标准化.
    if normal:
        diagnosis_samples = scalar_stand(diagnosis_samples)
    else:  # 需要做一个数据转换，转换成np格式.
        diagnosis_samples = np.asarray(diagnosis_samples)

    print(diagnosis_samples.shape)
    return diagnosis_samples

#深度学习传入参数
def parse_args_DL():
    parser = argparse.ArgumentParser(description='Train')

    # model and data parameters
    parser.add_argument('--model_name', type=str, default='CNN', help='the name of the model')
    parser.add_argument('--data_name', type=str, default='ScreedPressure', help='the name of the data')
    parser.add_argument('--data_dir', type=str, default='D:\IFD\data\ScreedPressure', help='the directory of the data')
    parser.add_argument("--data_length", type=int, default=128, help="size of each data")
    parser.add_argument("--step_length", type=int, default=4, help="size of step")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument('--transfer_task', type=list, default=[[0], [1]], help='transfer learning tasks')
    parser.add_argument('--normlizetype', type=str, default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=16, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')


    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-3, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='step', help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--tradeoff', type=list, default=[1, 'exp'], help='coefficients of loss')
    parser.add_argument('--steps', type=str, default='150, 250', help='the learning rate decay for step and stepLR')


    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=2, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=600, help='the interval of log training information')

    args = parser.parse_args()
    return args


#迁移学习(SDA,MDA,MDG)传入参数
def parse_args_DTL():
    parser = argparse.ArgumentParser(description='Train')

    # model and data parameters
    parser.add_argument('--area', type=str, default='MDG', choices=['SDA','MDA','MDG'], help='the name of the method')
    parser.add_argument('--model_name', type=str, default='DG',choices=['DAN','MFSAN', 'MSSA', 'MINE','DG'], help='the name of the model')
    parser.add_argument('--source_name', type=list, default=['SendLineAux_0','SendLineAux_1','SendLineAux_2','SendLineAux_4','SendLineAux_5'],help='the name of the source data')
    parser.add_argument('--target_name', type=str, default='SendLineAux_3', help='the name of the target data')
    parser.add_argument('--data_dir', type=str, default='D:\IFD\data', help='the directory of the data')
    parser.add_argument('--train_mode', type=str, choices=['single_source', 'source_combine', 'multi_source'],
                        default="multi_source", help='the mode for training')
    parser.add_argument("--data_length", type=int, default=128, help="size of each data")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument('--num_classes', type=int, default=2, help='the classes of data')
    parser.add_argument('--normlizetype', type=str,choices=['0-1', '-1-1', 'mean-std'], default='mean-std', help='nomalization type')

    # training parameters
    parser.add_argument('--device', type=str, default='cpu', help='device')
    parser.add_argument('--cuda_device', type=str, default='0', help='assign device')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='the directory to save the model')
    parser.add_argument("--pretrained", type=bool, default=False, help='whether to load the pretrained model')
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize of the training process')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of training process')

    # optimization information
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='adam', help='the optimizer')
    parser.add_argument('--lr', type=float, default=1e-5, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum for sgd')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='the learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate scheduler parameter for step and exp')
    parser.add_argument('--tradeoff', type=list, default=[1, 'exp'], help='coefficients of loss')
    parser.add_argument('--steps', type=str, default='100', help='the learning rate decay for step and stepLR')

    # save, load and display information
    parser.add_argument('--max_epoch', type=int, default=2, help='max number of epoch')
    parser.add_argument('--print_step', type=int, default=600, help='the interval of log training information')
    args = parser.parse_args()
    return args

app = QApplication([])
main = MainWindow()
main.ui.show()
app.exec_()



