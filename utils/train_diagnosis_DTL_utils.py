#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
@Time     : 2023/04/22 14:11
@Author   : Xiaorong Zheng
@Version  : V1
@File     : train_diagnosis_DTL_utils.py
@Software : PyCharm
"""

# Local Modules
import logging
import os
import time
import warnings
from collections import defaultdict
from tqdm import tqdm
import itertools

# Third-party Modules
import math
import torch
from torch import nn
from torch import optim

# Self-written Modules
import diagnosis_models
import datasets

class train_diagnosis_DTL_utils(object):

    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

        # Consider the gpu or cpu condition
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        model = getattr(diagnosis_models, args.model_name)
        self.model = model(in_channel=1, num_classes=args.num_classes)

        if args.train_mode == 'source_combine':
            self.num_source = 1
        else:
            self.num_source = len(args.source_name)

    def _get_lr_scheduler(self, optimizer):
        '''
        Get learning rate scheduler for optimizer.
        '''
        args = self.args

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            lr_scheduler = None
        else:
            raise Exception("lr schedule not implemented")

        return lr_scheduler

    def _get_optimizer(self, model):
        '''
        Get optimizer for model.
        '''
        args = self.args
        if type(model) == list:
            par = filter(lambda p: p.requires_grad, itertools.chain(*map(list,
                                                                         [md.parameters() for md in model])))
        else:
            par = model.parameters()

        # Define the optimizer
        if args.opt == 'sgd':
            optimizer = optim.SGD(par, lr=args.lr, momentum=args.momentum,
                                  weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            optimizer = optim.Adam(par, lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implemented")

        return optimizer

    def _get_tradeoff(self, tradeoff_list, epoch):
        '''
        Get trade-off parameters for loss.
        '''
        tradeoff = []
        for item in tradeoff_list:
            if item == 'exp':
                tradeoff.append(2 / (1 + math.exp(-10 * epoch / self.args.max_epoch)) - 1)
            else:
                tradeoff.append(item)

        return tradeoff

    def get_next_batch(self,loaders, iters, src, device):
        inputs, labels = None, None
        if type(src) == list:#多个域
            for key in src:
                try:
                    inputs, labels = next(iters[key])
                    break
                except StopIteration:
                    continue
            if inputs == None:
                for key in src:
                    iters[key] = iter(loaders[key])
                inputs, labels = next(iters[src[0]])
        else:#单个域
            try:
                inputs, labels = next(iters[src])
            except StopIteration:
                iters[src] = iter(loaders[src])
                inputs, labels = next(iters[src])

        return inputs.to(device), labels.to(device)

    def _get_accuracy(self, preds, targets):
        assert preds.shape[0] == targets.shape[0]
        correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
        accuracy = correct / preds.shape[0]

        return accuracy

    def _init_data(self):
        '''
        Initialize the datasets.
        '''
        args = self.args

        self.datasets = {}

        #args.data_dir = D:/IFD/data/Tension/op_0/fault
        # 获得数据所在的文件夹  D:/IFD/data/Tension/op_0/fault --> D:/IFD/data/Tension/op_0 --> D:/IFD/data/Tension
        data_path = args.data_dir.rsplit("/", 1)[0]

        #源域数据
        for source in args.source_name:  # 'Tension_1', 'Tension_2'
            if '_' in source:
                src, op = source.split('_')[0], source.split('_')[1]
                try:
                    Dataset = getattr(datasets, '%s' % src)  # 执行Tension.py
                except:
                    raise Exception("data name type not implemented")
                self.datasets[source] = Dataset(data_path, args.normlizetype,
                                                op=op,signal_size=args.data_length).data_preprare(is_src=True)  # 源域  self.datasets[source]
        for key in self.datasets.keys():
            logging.info('source set {} length {}.'.format(key, len(self.datasets[key])))
            self.datasets[key].summary()

        #目标域数据
        if args.area == 'SDA' or args.area == 'MDA':#SDA和MDA目标域分为训练集和验证集
            if '_' in args.target_name:
                tgt, op = args.target_name.split('_')[0], args.target_name.split('_')[1]
                try:
                    Dataset = getattr(datasets, '%s' % tgt)
                except:
                    raise Exception("data name type not implemented")
                self.datasets['train'], self.datasets['val'] = Dataset(data_path, args.normlizetype,
                                                                   op=op).data_preprare()
            logging.info('target training set length {}, target validation set length {}.'.format(
                len(self.datasets['train']), len(self.datasets['val'])))
            self.datasets['train'].summary();self.datasets['val'].summary()

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           batch_size=args.batch_size,
                                                           shuffle=(False if x == 'val' else True),
                                                           num_workers=args.num_workers,
                                                           drop_last=(False if x == 'val' else True),
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in (['train', 'val'] + args.source_name)}
            self.iters = {x: iter(self.dataloaders[x]) for x in (['train', 'val'] + args.source_name)}

            # drop_last=(False if x == 'val' else True),

        elif args.area == 'MDG':#MDG目标域全部作为验证集
            if '_' in args.target_name:
                tgt, op = args.target_name.split('_')[0], args.target_name.split('_')[1]
                try:
                    Dataset = getattr(datasets, '%s' % tgt)
                except:
                    raise Exception("data name type not implemented")
                self.datasets['val'] = Dataset(data_path, args.normlizetype,
                                                                   op=op).data_preprare(is_src=True)
            logging.info('target validation set length {}.'.format(len(self.datasets['val'])))
            self.datasets['val'].summary()

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                           batch_size=args.batch_size,
                                                           shuffle=(False if x == 'val' else True),
                                                           num_workers=args.num_workers,
                                                           drop_last=(False if x == 'val' else True),
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in (['val'] + args.source_name)}
            self.iters = {x: iter(self.dataloaders[x]) for x in (['val'] + args.source_name)}
            #drop_last=(False if x == 'val' else True),

        return self.dataloaders['val']

    def _get_next_batch(self,loaders, iters, src, device):
        inputs, labels = None, None
        if type(src) == list:
            for key in src:
                try:
                    inputs, labels = next(iters[key])
                    break
                except StopIteration:
                    continue
            if inputs == None:
                for key in src:
                    iters[key] = iter(loaders[key])
                inputs, labels = next(iters[src[0]])
        else:
            try:
                inputs, labels = next(iters[src])
            except StopIteration:
                iters[src] = iter(loaders[src])
                inputs, labels = next(iters[src])

        return inputs.to(device), labels.to(device)

    def _get_accuracy(self, preds, targets):
        assert preds.shape[0] == targets.shape[0]
        correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
        accuracy = correct / preds.shape[0]

        return accuracy

    def train_1src_and_srccomb(self):
        args = self.args

        model = getattr(diagnosis_models, args.model_name)

        if args.model_name in ['CNN', 'WDCNN']:
            training_mode = 0
            self.model = model(in_channel=1, num_classes=args.num_classes)
        elif args.model_name in ['DAN', 'CDAN', 'ACDANN']:
            training_mode = 1
            self.model = model(in_channel=2, num_classes=args.num_classes)
        elif args.model_name in ['DANN']:
            training_mode = 2
            self.model = model(in_channel=1, num_classes=args.num_classes)
        elif args.model_name in ['ADACL']:
            training_mode = 3
            self.model = model(in_channel=1, num_classes=args.num_classes, num_source=1)
        elif args.model_name in ['MFSAN']:
            training_mode = 4
            self.model = model(in_channel=1, num_classes=args.num_classes, num_source=1)
        elif args.model_name in ['MSSA']:
            training_mode = 5
            self.model = model(in_channel=1, num_classes=args.num_classes, num_source=1)
        elif args.model_name in ['MADN']:
            training_mode = 6
            self.model = model(args.lr, in_channel=1, num_classes=args.num_classes, num_source=1)
        else:
            raise Exception("model type not implemented")

        if args.train_mode == 'single_source':#单源域适应
            src = args.source_name[0]
        elif args.train_mode == 'source_combine':#多源域结合成单源域再适应
            src = args.source_name

        self.model = self.model.to(self.device)
        self._init_data()
        if training_mode == 6:
            self.optimizer = self._get_optimizer([self.model.encoder, self.model.clf,
                                                  self.model.discriminator])
        else:
            self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0
        best_epoch = 0

        loss_list = []  # 训练阶段损失
        val_acc_list = []  # 验证阶段目标域准确率

        for epoch in range(args.max_epoch + 1):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase
            for phase in ['train', 'val']:
                epoch_acc = 0
                epoch_loss_all = 0.0
                epoch_loss = defaultdict(float)  # 初始化为0.0

                # Set model to train mode or evaluate mode
                if phase == 'train':
                    self.model.train()
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch)
                else:
                    self.model.eval()

                num_iter = len(self.iters[phase])       #以目标域训练集的迭代次数
                if args.train_mode == 'source_combine':
                    num_iter *= len(src)
                for i in tqdm(range(num_iter), ascii=True):
                    if phase == 'train' or training_mode == 5:  #MSSA模型的验证阶段需要源域和目标域的相似度加权
                        source_data, source_labels = self._get_next_batch(self.dataloaders,
                                                                          self.iters, src, self.device)
                    target_data, target_labels = self._get_next_batch(self.dataloaders,
                                                                      self.iters, phase, self.device)
                    if phase == 'train':
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            if training_mode == 0:
                                pred, loss = self.model(target_data,source_data, source_labels)
                                epoch_loss[0] += loss
                            elif training_mode == 1:
                                logging.info("{},{}".format(source_data.shape,source_labels.shape))
                                pred, loss_0, loss_1 = self.model(target_data,
                                                                  source_data, source_labels)
                                loss = loss_0 + tradeoff[0] * loss_1
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1
                            elif training_mode == 2:
                                pred, loss_0, loss_1, acc_d = self.model(target_data,
                                                                         source_data, source_labels)
                                loss = loss_0 + tradeoff[0] * loss_1
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1
                            elif training_mode == 3:
                                pred, loss_0, loss_1, loss_2 = self.model(target_data,
                                                                          self.device, source_data, source_labels,
                                                                          source_idx=0)
                                loss = loss_0 + tradeoff[0] * loss_1 + \
                                       tradeoff[1] * loss_2
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1
                                epoch_loss[2] += loss_2
                            elif training_mode == 4:
                                pred, loss_0, loss_1, loss_2 = self.model(target_data,
                                                                          source_data, source_labels, source_idx=0)
                                loss = loss_0 + tradeoff[0] * loss_1 + \
                                       tradeoff[1] * loss_2
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1
                                epoch_loss[2] += loss_2
                            elif training_mode == 5:
                                pred, loss_0, loss_1 = self.model(target_data,
                                                                  self.device, [source_data], [source_labels])
                                loss = loss_0 + tradeoff[0] * loss_1
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1
                                logging.info("{}, {}".format(loss_0,loss_1))
                            elif training_mode == 6:
                                pred, loss_0, loss_1 = self.model(target_data,
                                                                  self.device, [source_data], [source_labels],
                                                                  rec=(i < num_iter / 2))
                                loss = loss_0 + tradeoff[0] * loss_1
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1

                            epoch_loss_all += loss.item()

                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            if training_mode in [5, 6]:
                                pred = self.model(target_data, self.device,
                                                  [source_data], [source_labels])
                            elif training_mode in [3, 4]:
                                pred = self.model(target_data, device=self.device)
                            else:
                                pred = self.model(target_data)

                    epoch_acc += self._get_accuracy(pred, target_labels)

                # Print the train and val information via each epoch
                epoch_acc = epoch_acc / num_iter
                if phase == 'train':
                    for key in epoch_loss.keys():
                        logging.info('{}-Loss_{}: {:.4f}'.format(phase, key,
                                                                 epoch_loss[key] / num_iter))
                    logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))
                    loss_list.append(epoch_loss_all / num_iter)  # 训练阶段有损失
                else:
                    logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))

                    # log the best model according to the val accuracy
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch,
                                                                            best_acc))
                    val_acc_list.append(epoch_acc)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, loss_list, val_acc_list

    def train_multi_src(self):
        args = self.args

        model = getattr(diagnosis_models, args.model_name)


        if args.model_name in ['MFSAN']:
            training_mode = 1
            self.model = model(in_channel=1, num_classes=args.num_classes, num_source=self.num_source)
        elif args.model_name in ['MADN', 'MSSA','MINE']:
            training_mode = 2
            self.model = model(in_channel=1, num_classes=args.num_classes, num_source=self.num_source)
        elif args.model_name in ['DG']:#泛化模型
            training_mode = 3
            self.model = model(in_channel=1, num_classes=args.num_classes, num_source=self.num_source)
        else:
            raise Exception("model type not implemented")

        self._init_data()  # 加载数据(DA DG)
        self.model = self.model.to(self.device)
        self.optimizer = self._get_optimizer(self.model)
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        best_acc = 0.0
        best_epoch = 0

        loss_list = []#训练阶段损
        val_acc_list = []#验证阶段目标域准确率

        for epoch in range(args.max_epoch):

            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))

            # Each epoch has a training and val phase(DA的train是指目标域训练集，DG的train是指用源域训练,iters['train'])
            for phase in ['train', 'val']:
                epoch_acc = 0
                epoch_loss_all = 0.0
                val_epoch_loss_all = 0.0
                epoch_loss = defaultdict(float)  # 初始化为0.0

                # Set model to train mode or evaluate mode
                if phase == 'train':
                    self.model.train()
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch)
                else:
                    self.model.eval()

                #确定迭代次数
                if args.area == 'MDA':
                    if phase == 'train':
                        num_iter = 0
                        for x in (args.source_name):
                            num_iter_ = len(self.iters[x])  # 以样本数最多的源域迭代次数
                            if num_iter_ > num_iter:
                                num_iter = num_iter_
                    else:
                        num_iter = len(self.iters[phase])  # 以目标域迭代次数

                    if phase == 'train' and (training_mode == 0 or training_mode == 1):
                        num_iter *= len(args.source_name)  # 目标域迭代次数=原来的迭代次数*源域个数（每个源域训练时都要迭代完目标域）

                elif args.area == 'MDG':
                    if phase == 'train':
                        num_iter=0
                        for x in (args.source_name):
                            num_iter_ = len(self.iters[x])# 以样本数最多的源域迭代次数
                            if num_iter_>num_iter:
                                num_iter = num_iter_
                    else:
                        num_iter = len(self.iters[phase])

                for i in tqdm(range(num_iter), ascii=True):
                    #确定输入数据
                    if args.area == 'MDA':
                        #两个phase都需要目标域
                        target_data, target_labels = self._get_next_batch(self.dataloaders, self.iters, phase,self.device)  # phase=train or val
                        if phase == 'train':
                            if training_mode != 2:
                                idx = i % self.num_source  # idx表输入哪个源域，mode=0或者1时用到，只有一个特征提取器轮流输入source
                                source_data, source_labels = self._get_next_batch(self.dataloaders,self.iters, args.source_name[idx],
                                                                          self.device)
                            elif training_mode == 2:
                                source_data, source_labels = [], []
                                for src in args.source_name:
                                    data_item, labels_item = self._get_next_batch(self.dataloaders,
                                                                          self.iters, src, self.device)
                                    source_data.append(data_item)
                                    source_labels.append(labels_item)

                    elif args.area == 'MDG':
                        if phase == 'train':
                            if training_mode == 0 or training_mode == 1:
                                idx = i % self.num_source  # idx表输入哪个源域，mode=0或者1时用到，只有一个特征提取器轮流输入source
                                source_data, source_labels = self._get_next_batch(self.dataloaders, self.iters,
                                                                                      args.source_name[idx],
                                                                                      self.device)
                            else:
                                source_data, source_labels = [], []#列表长度等于源域个数
                                for src in args.source_name:
                                    data_item, labels_item = self._get_next_batch(self.dataloaders,
                                                                                      self.iters, src, self.device)
                                    source_data.append(data_item)
                                    source_labels.append(labels_item)
                        elif phase == 'val':  # DG只有val阶段有target_data
                            target_data, target_labels = self._get_next_batch(self.dataloaders, self.iters, phase,self.device)

                    if phase == 'train':
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            if training_mode == 0:  # ADACL
                                pred, loss_0, loss_1, loss_2 = self.model(target_data,
                                  self.device, source_data, source_labels, source_idx=idx)
                                loss = loss_0 + tradeoff[0] * loss_1 + tradeoff[1] * loss_2
                                epoch_loss[0] += loss_0;epoch_loss[1] += loss_1
                                epoch_loss[2] += loss_2  # 源域分类损失，领域判别器损失，分类器差异损失

                            elif training_mode == 1:  # MFSAN
                                pred, loss_0, loss_1, loss_2 = self.model(target_data,source_data, source_labels, source_idx=idx)
                                loss = loss_0 + tradeoff[0] * loss_1 + tradeoff[0] * loss_2
                                epoch_loss[0] += loss_0;epoch_loss[1] += loss_1
                                epoch_loss[2] += loss_2  # 源域分类损失，mmd损失，分类器差异损失

                            elif training_mode == 2:  # MADN,MSSA,MINE
                                pred, loss_0, loss_1 = self.model(target_data,device=self.device,source_data=source_data,source_label=source_labels)
                                loss = loss_0 + tradeoff[0] * loss_1

                                #打印信息用
                                epoch_loss[0] += loss_0;epoch_loss[1] += loss_1  # 源域分类损失，特征差异损失


                            elif training_mode == 3:  #泛化模型
                                loss_0, loss_1 = self.model(device=self.device, source_data=source_data, source_label=source_labels)  # 训练阶段目标域不参与，所以没有预测结果
                                loss = loss_0 + tradeoff[1] * loss_1
                                epoch_loss[0] += loss_0;
                                epoch_loss[1] += loss_1  # 源域分类损失，迁移损失



                            epoch_loss_all+=loss.item()

                            # backward
                            loss.backward()
                            self.optimizer.step()
                    else:
                        with torch.no_grad():
                            if training_mode == 2:
                                #pred = self.model(target_data, self.device,source_data, source_labels)##需要源域相似度加权分类器
                                pred = self.model(target_data, device=self.device)
                            else:
                                pred = self.model(data_tgt=target_data, device=self.device)


                    if args.area == 'MDA':
                        epoch_acc += self._get_accuracy(pred, target_labels)
                    elif args.area == 'MDG' and phase == 'val':  # DG只有验证阶段有目标域数据输出测试结果
                        epoch_acc += self._get_accuracy(pred, target_labels)

                # Print the train and val information via each epoch
                if args.area == 'MDA':
                    epoch_acc = epoch_acc / num_iter
                    if phase == 'train':

                        for key in epoch_loss.keys():
                            logging.info('{}-Loss_{}: {:.4f}'.format(phase, key,
                                                                     epoch_loss[key] / num_iter))
                        logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))
                        loss_list.append(epoch_loss_all / num_iter)  # 训练阶段有损失
                    else:

                        for key in epoch_loss.keys():
                            logging.info('{}-Loss_{}: {:.4f}'.format(phase, key,
                                                                     epoch_loss[key] / num_iter))

                        logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))
                        val_acc_list.append(epoch_acc)

                        # log the best model according to the val accuracy
                        if epoch_acc >= best_acc:
                            best_acc = epoch_acc
                            best_epoch = epoch

                        logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch,
                                                                                      best_acc))



                elif args.area == 'MDG':  # DG的训练阶段输出loss，验证阶段输出准确率
                    if phase == 'train':
                        for key in epoch_loss.keys():
                            logging.info('{}-Loss_{}: {:.4f}'.format(phase, key,
                                                                     epoch_loss[key] / num_iter))

                        loss_list.append(epoch_loss_all / num_iter)  # 训练阶段有损失
                    else:
                        epoch_acc = epoch_acc / num_iter
                        logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))

                        val_acc_list.append(epoch_acc)

                        # log the best model according to the val accuracy
                        if epoch_acc >= best_acc:
                            best_acc = epoch_acc
                            best_epoch = epoch
                        logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch,
                                                                                      best_acc))
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, loss_list, val_acc_list

