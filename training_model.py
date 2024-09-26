#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd.variable import Variable

import diagnosis_models
from sklearn import preprocessing  # 0-1编码
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Activation, Flatten, LSTM, Dropout, GRU
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm, metrics
from sklearn.multiclass import OneVsRestClassifier

from feature_extraction import feature_extraction
from sklearn.metrics import accuracy_score


def training_with_svm(X_train, y_train, X_test, y_test):
    '''
    使用 SVM 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
            clf_rfc：训练完成的模型
            score：模型在验证集上的得分
            X_train_feature_extraction：将原数据进行了特征提取过的训练集
            X_test_feature_extraction：将原数据进行了特征提取过的测试集
    '''

    loader = np.empty(shape=[X_train.shape[0], 16])#16个统计学特征
    for i in range(X_train.shape[0]):
        loader[i] = feature_extraction(X_train[i])
    X_train_feature_extraction = loader

    loader = np.empty(shape=[X_test.shape[0], 16])
    for i in range(X_test.shape[0]):
        loader[i] = feature_extraction(X_test[i])
    X_test_feature_extraction = loader

    clf_rfc = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=0))
    clf_rfc.fit(X_train_feature_extraction, y_train)
    score = clf_rfc.score(X_test_feature_extraction, y_test)
    return clf_rfc, score, X_train_feature_extraction, X_test_feature_extraction

def accuracy(y_pred,y_true):
    y_pred_cls = torch.argmax(nn.Softmax(dim=1)(y_pred),dim=1).data
    return accuracy_score(y_true.cpu().numpy(),y_pred_cls.cpu().numpy())

def one_hot(y_train, y_valid_test):
    '''
    函数说明：one-hot编码
                将样本标签编码为 含有 10个 元素的列表：[1，0，0，0，0，0，0，0，0，0]
                分别对应文件夹中的十个文件，如，第一个文件是一个 故障数据 的文件，其故障对应到列表的第一个元素（第一个元素为1）
                所以，这就是一个10分类问题，即最终要找出故障的位置

    Parameters:
        y_train : 训练集标签
        y_valid_test : <验证和测试集>标签
    Returns:
        y_train : 编码后的训练集标签
        y_valid_test : 编码后的<验证和测试集>标签
    '''
    y_train = np.array(y_train).reshape([-1, 1])
    y_valid_test = np.array(y_valid_test).reshape([-1, 1])

    Encoder = preprocessing.OneHotEncoder()
    Encoder.fit(y_train)  # 因为 y_train 和 y_valid_test 的类别相同，所以在其中一个上面fit就可以了

    y_train = Encoder.transform(y_train).toarray()
    y_valid_test = Encoder.transform(y_valid_test).toarray()

    y_train = np.asarray(y_train, dtype=np.int32)
    y_valid_test = np.asarray(y_valid_test, dtype=np.int32)
    return y_train, y_valid_test

def training_with_1D_CNN(X_train, y_train, X_test, y_test, batch_size=128, epochs=20, num_classes=10):
    '''
    使用 1D_CNN 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模性训练的 批次大小
    :param epochs: 模性训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模性训练(fit)的返回参数
            score：模型在验证集上的得分
    '''
    X_train, X_test = X_train[:, :, np.newaxis], X_test[:, :, np.newaxis]  # 添加一个新的维度
    # 输入数据的维度
    input_shape = X_train.shape[1:]
    y_train, y_test = one_hot(y_train, y_test)
    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    # 实例化一个Sequential
    model = Sequential()

    # 第一层卷积
    model.add(Conv1D(filters=32, kernel_size=20, strides=8, padding='same', kernel_regularizer=l2(1e-4),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=4, strides=4, padding='valid'))
    # 从卷积到全连接需要展平
    model.add(Flatten())
    # 添加全连接层
    model.add(Dense(units=100, activation='relu', kernel_regularizer=l2(1e-4)))
    # 增加输出层，共num_classes个单元
    model.add(Dense(units=num_classes, activation='softmax', kernel_regularizer=l2(1e-4)))

    # 编译模型
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    # 开始模型训练
    history = model.fit(X_train, y_train, batch_size=128, epochs=epochs, verbose=1,
                        validation_split=0.25, shuffle=True)
    # 评估模型
    score = model.evaluate(X_test, y_test, verbose=0)

    return model, history, score


def training_with_LSTM(X_train, y_train, X_test, y_test, batch_size=128, epochs=60, num_classes=10):
    '''
    使用 LSTM 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模性训练的 批次大小
    :param epochs: 模性训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模性训练(fit)的返回参数
            score：模型在验证集上的得分
    '''
    X_train, X_test = X_train[:, np.newaxis, :], X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 输入数据的维度
    input_shape = X_train.shape[1:]
    y_train, y_test = one_hot(y_train, y_test)

    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    model_LSTM = Sequential()
    # LSTM 第一层
    model_LSTM.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model_LSTM.add(Dropout(0.5))
    # LSTM 第二层
    model_LSTM.add(LSTM(128, return_sequences=True))
    model_LSTM.add(Dropout(0.5))
    # LSTM 第三层
    model_LSTM.add(LSTM(256))
    model_LSTM.add(Dropout(0.5))
    # Dense层
    model_LSTM.add(Dense(num_classes, activation='sigmoid'))

    # 编译模型
    model_LSTM.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    # 开始模型训练
    history = model_LSTM.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_split=0.25, shuffle=True)
    # 评估模型
    score = model_LSTM.evaluate(X_test, y_test, verbose=0)

    return model_LSTM, history, score


def training_with_GRU(X_train, y_train, X_test, y_test, batch_size=128, epochs=60, num_classes=10):
    '''
    使用 GRU 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :param batch_size: 模性训练的 批次大小
    :param epochs: 模性训练的轮数
    :param num_classes: 分类数
    :return:
            model：训练完成的模型
            history：模性训练(fit)的返回参数
            score：模型在验证集上的得分
    '''
    X_train, X_test = X_train[:, np.newaxis, :], X_test[:, np.newaxis, :]  # 添加一个新的维度
    # 输入数据的维度
    input_shape = X_train.shape[1:]
    y_train, y_test = one_hot(y_train, y_test)

    K.clear_session()  # 清除会话，否则当执行完一个神经网络，接着执行下一个神经网络时可能会会报错

    model_GRU = Sequential()
    model_GRU.add(GRU(64, return_sequences=True, input_shape=input_shape, activation='tanh'))
    model_GRU.add(Dropout(0.5))
    model_GRU.add(GRU(128, activation='tanh'))
    model_GRU.add(Dropout(0.5))
    model_GRU.add(Dense(num_classes, activation='sigmoid'))

    # 编译模型
    model_GRU.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # model.summary()

    # 开始模型训练
    history = model_GRU.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
                             validation_split=0.25, shuffle=True)
    # 评估模型
    score = model_GRU.evaluate(X_test, y_test, verbose=0)

    return model_GRU, history, score


def training_with_random_forest(X_train, y_train, X_test, y_test):
    '''
    使用 随机森林 进行训练
    :param X_train: 训练集
    :param y_train: 训练集标签
    :param X_valid: 验证集
    :param y_valid: 验证集标签
    :param X_test: 测试集
    :param y_test: 测试集标签
    :return:
            clf_rfc：训练完成的模型
            score：模型在验证集上的得分
            X_train_feature_extraction：将原数据进行了特征提取过的训练集
            X_test_feature_extraction：将原数据进行了特征提取过的测试集
    '''

    loader = np.empty(shape=[X_train.shape[0], 16])
    for i in range(X_train.shape[0]):
        loader[i] = feature_extraction(X_train[i])
    X_train_feature_extraction = loader

    loader = np.empty(shape=[X_test.shape[0], 16])
    for i in range(X_test.shape[0]):
        loader[i] = feature_extraction(X_test[i])
    X_test_feature_extraction = loader

    clf_rfc = RandomForestClassifier(n_estimators=17, max_depth=21, criterion='gini', min_samples_split=2,
                                       max_features=9, random_state=60 )
    clf_rfc.fit(X_train_feature_extraction, y_train)
    score = clf_rfc.score(X_test_feature_extraction, y_test)
    return clf_rfc, score, X_train_feature_extraction, X_test_feature_extraction

