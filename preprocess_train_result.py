#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import torch
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import scikitplot as skplt
import torch.nn.functional as nnf



# 绘制混淆矩阵
def my_confusion_matrix(preds, labels, conf_matrix):
    preds = torch.argmax(preds, 1)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix

def dtl_plot_history_curcvs(loss_list, val_acc_list, save_path, model_name):
    epochs = range(len(loss_list))
    plt.plot(epochs, val_acc_list, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_acc.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()  # 再画一个图，显式 创建figure对象
    plt.plot(epochs, loss_list, 'bo', label='Training loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_loss.png', dpi=150, bbox_inches='tight')
    plt.close()


def dtl_plot_confusion_matrix(model, model_name, save_path, val_dataloader):
    '''
    绘制混淆矩阵
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''

    conf_matrix = torch.zeros(2, 2)
    for data, target in val_dataloader:
        output = model(data.to(torch.device("cuda")))
        conf_matrix = my_confusion_matrix(output, target, conf_matrix)

    logging.info('{}'.format(conf_matrix))

    con_mat = conf_matrix.cpu().numpy()
    # 绘制混淆矩阵
    #由于故障样本少，所以不做归一化处理
    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)  # np.around(): 四舍五入

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm,
                annot=True,  # annot: 默认为False，为True的话，会在格子上显示数字
                cmap='Blues'  # 热力图颜色
                )

    labels = ['Fault', 'Healthy']
    xlocations = np.array(range(len(labels)))
    plt.xticks([index + 0.5 for index in xlocations], labels, rotation='horizontal')
    plt.yticks([index + 0.5 for index in xlocations], labels)

    #plt.ylim(0, 2)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(save_path + '/' + model_name + '_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()

def   dtl_brief_classification_report(model, model_name, val_dataloader):
    '''
    计算 分类报告
    :param model: 模型
    :param model_name:  模型名称
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return: classification_report：分类报告
    '''
    y_preds_list = []
    y_test_list = []
    for data, target in val_dataloader:
        logits = model(data.to(torch.device("cuda")))
        pred = logits.argmax(dim=1)
        pred.unsqueeze_(1)
        target.unsqueeze_(1)
        y_preds_list.append(pred)
        y_test_list.append(target)

    y_preds = torch.cat(y_preds_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    y_preds = y_preds.cpu().numpy()
    y_test = y_test.cpu().numpy()
    classification_report = metrics.classification_report(y_test, y_preds)

    return classification_report


def dtl_plot_metrics(model, model_name, save_path, val_dataloader):
    '''
    绘制 ROC曲线 和 精度召回曲线
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    y_probas_list = []
    y_test_list = []
    for data, target in val_dataloader:
        logits = model(data.to(torch.device("cuda")))
        probas = nnf.softmax(logits, dim=1)
        probas.unsqueeze_(1)
        target.unsqueeze_(1)
        y_probas_list.append(probas)
        y_test_list.append(target)
    y_probas = torch.cat(y_probas_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    y_probas = y_probas.detach().cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_probas = np.squeeze(y_probas, 1)
    y_test = np.squeeze(y_test, 1)
    print(y_probas.shape, y_test.shape)
    logging.info('{}{}'.format(y_probas.shape, y_test.shape))       #y_test验证集,y_probas模型输出

    # 绘制“ROC曲线”
    skplt.metrics.plot_roc(y_test, y_probas, title=model_name+' ROC Curves', figsize=(7, 7),
                           # title_fontsize = 24, text_fontsize = 16
                           )
    plt.savefig(save_path + '/' + model_name + '_ROC_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制“精度召回曲线”
    skplt.metrics.plot_precision_recall(y_test, y_probas, title=model_name+' Precision-Recall Curves', figsize=(7, 7),
                                        # title_fontsize = 24, text_fontsize = 16
                                        )
    plt.savefig(save_path + '/' + model_name + '_Precision_Recall_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()


def dl_plot_history_curcvs(loss_list, val_loss_list, acc_list, val_acc_list, save_path, model_name):
    epochs = range(len(loss_list))
    plt.plot(epochs, acc_list, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_list, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_acc.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()  # 再画一个图，显式 创建figure对象
    plt.plot(epochs, loss_list, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_list, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

def dl_plot_confusion_matrix(model, model_name, save_path, val_dataloader):
    '''
    绘制混淆矩阵
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''

    conf_matrix = torch.zeros(2, 2)
    for data, target in val_dataloader:
        output = model(data.to(torch.device("cuda")))
        conf_matrix = my_confusion_matrix(output, target, conf_matrix)

    con_mat = conf_matrix.cpu().numpy()
    # 绘制混淆矩阵
    #con_mat = confusion_matrix(y_test, y_preds)

    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)  # np.around(): 四舍五入

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm,
                annot=True,  # annot: 默认为False，为True的话，会在格子上显示数字
                cmap='Blues'  # 热力图颜色
                )

    plt.ylim(0, 2)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(save_path + '/' + model_name + '_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    #plt.show()

def   dl_brief_classification_report(model, model_name, val_dataloader):
    '''
    计算 分类报告
    :param model: 模型
    :param model_name:  模型名称
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return: classification_report：分类报告
    '''
    y_preds_list = []
    y_test_list = []
    for data, target in val_dataloader:
        logits = model(data.to(torch.device("cuda")))
        pred = logits.argmax(dim=1)
        pred.unsqueeze_(1)
        target.unsqueeze_(1)
        y_preds_list.append(pred)
        y_test_list.append(target)

    y_preds = torch.cat(y_preds_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    y_preds = y_preds.cpu().numpy()
    y_test = y_test.cpu().numpy()
    classification_report = metrics.classification_report(y_test, y_preds)

    return classification_report


def dl_plot_metrics(model, model_name, save_path, val_dataloader):
    '''
    绘制 ROC曲线 和 精度召回曲线
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    y_probas_list = []
    y_test_list = []
    for data, target in val_dataloader:
        logits = model(data.to(torch.device("cuda")))
        probas = nnf.softmax(logits, dim=1)
        probas.unsqueeze_(1)
        target.unsqueeze_(1)
        y_probas_list.append(probas)
        y_test_list.append(target)
    y_probas = torch.cat(y_probas_list, dim=0)
    y_test = torch.cat(y_test_list, dim=0)
    y_probas = y_probas.detach().cpu().numpy()
    y_test = y_test.cpu().numpy()
    y_probas = np.squeeze(y_probas, 1)
    y_test = np.squeeze(y_test, 1)
    print(y_probas.shape, y_test.shape)

    # 绘制“ROC曲线”
    skplt.metrics.plot_roc(y_test, y_probas, title=model_name+' ROC Curves', figsize=(7, 7),
                           # title_fontsize = 24, text_fontsize = 16
                           )
    plt.savefig(save_path + '/' + model_name + '_ROC_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制“精度召回曲线”
    skplt.metrics.plot_precision_recall(y_test, y_probas, title=model_name+' Precision-Recall Curves', figsize=(7, 7),
                                        # title_fontsize = 24, text_fontsize = 16
                                        )
    plt.savefig(save_path + '/' + model_name + '_Precision_Recall_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_history_curcvs(history, save_path, model_name):
    '''
    绘制 训练集 和 验证集 的 损失 及 正确率 曲线
    :param history: 模型训练（fit)的返回参数
    :param save_path: 生成图片的保存路径
    :param model_name: 模型名称
    :return:
    '''
    acc = history.history['accuracy']  # 每一轮 在 训练集 上的 精度
    val_acc = history.history['val_accuracy']  # 每一轮 在 验证集 上的 精度
    loss = history.history['loss']  # 每一轮 在 训练集 上的 损失
    val_loss = history.history['val_loss']  # 每一轮 在 验证集 上的 损失

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    print(save_path)
    plt.savefig(save_path + '/' + model_name + '_train_valid_acc.png', dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()  # 再画一个图，显式 创建figure对象
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(save_path + '/' + model_name + '_train_valid_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()

def plot_confusion_matrix(model, model_name, save_path, X_test, y_test):
    '''
    绘制混淆矩阵
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    if '1D_CNN' == model_name:
        X_test = X_test[:, :, np.newaxis]  # 添加一个新的维度
    elif 'LSTM' == model_name or 'GRU' == model_name:
        X_test = X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 随机森林不需要添加维度

    # 这里两种的 预测函数 不同
    if 'random_forest' == model_name or 'SVM' == model_name:
        y_preds = model.predict(X_test)
    else:
        y_preds = model.predict_classes(X_test)

    #y_test = [np.argmax(item) for item in y_test]  # one-hot解码

    # 绘制混淆矩阵
    con_mat = confusion_matrix(y_test, y_preds)

    con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]  # 归一化
    con_mat_norm = np.around(con_mat_norm, decimals=2)  # np.around(): 四舍五入

    plt.figure(figsize=(8, 8))
    sns.heatmap(con_mat_norm,
                annot=True,  # annot: 默认为False，为True的话，会在格子上显示数字
                cmap='Blues'  # 热力图颜色
                )

    plt.ylim(0, 10)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.savefig(save_path + '/' + model_name + '_confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    # plt.show()


def brief_classification_report(model, model_name, X_test, y_test):
    '''
    计算 分类报告
    :param model: 模型
    :param model_name:  模型名称
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return: classification_report：分类报告
    '''
    if '1D_CNN' == model_name:
        X_test = X_test[:, :, np.newaxis]  # 添加一个新的维度
    elif 'LSTM' == model_name or 'GRU' == model_name:
        X_test = X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 随机森林不需要添加维度

    # 这里两种的 预测函数 不同
    if 'random_forest' == model_name or 'SVM' == model_name:
        y_preds = model.predict(X_test)
    else:
        y_preds = model.predict_classes(X_test)

    #y_test = [np.argmax(item) for item in y_test]  # one-hot解码
    classification_report = metrics.classification_report(y_test, y_preds)

    return classification_report


def plot_metrics(model, model_name, save_path, X_test, y_test):
    '''
    绘制 ROC曲线 和 精度召回曲线
    :param model: 模型
    :param model_name: 模型名称
    :param save_path: 生成图片的保存路径
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
    '''
    if '1D_CNN' == model_name:
        X_test = X_test[:, :, np.newaxis]  # 添加一个新的维度
    elif 'LSTM' == model_name or 'GRU' == model_name:
        X_test = X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 随机森林不需要添加维度

    y_probas = model.predict_proba(X_test)
    #y_test = [np.argmax(item) for item in y_test]  # one-hot解码
    # 绘制“ROC曲线”
    skplt.metrics.plot_roc(y_test, y_probas, title=model_name+' ROC Curves', figsize=(7, 7),
                           # title_fontsize = 24, text_fontsize = 16
                           )
    plt.savefig(save_path + '/' + model_name + '_ROC_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()

    # 绘制“精度召回曲线”
    skplt.metrics.plot_precision_recall(y_test, y_probas, title=model_name+' Precision-Recall Curves', figsize=(7, 7),
                                        # title_fontsize = 24, text_fontsize = 16
                                        )
    plt.savefig(save_path + '/' + model_name + '_Precision_Recall_Curves.png', dpi=150, bbox_inches='tight')
    # plt.show()
    plt.close()
