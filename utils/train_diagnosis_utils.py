# -*- coding: UTF-8 -*-
"""
@Time     : 2021/12/11 10:33
@Author   : Caiming Liu
@Version  : V1
@File     : train_diagnosis_utils.py.py
@Software : PyCharm
"""

# Local Modules
import logging
import os
import time
import warnings

# Third-party Modules
import torch
from torch import nn
from torch import optim
import numpy as np
# Self-written Modules
import diagnosis_models
import datasets

class train_diagnosis_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

        # Consider the gpu or cpu condition
        if torch.cuda.is_available() and args.device == "gpu":
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('using {} gpus'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "batch size should be divided by device count"
        else:
            warnings.warn("gpu is not available")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('using {} cpu'.format(self.device_count))

        # Load the datasets
        data_path = args.data_dir.rsplit("/", 1)[0]
        Dataset = getattr(datasets, args.data_name)  # 返回一个对象属性值
        self.datasets = {}

        # 源域数据
        for source in args.source_name:  # 'Tension_1', 'Tension_2'
            if '_' in source:
                src, op = source.split('_')[0], source.split('_')[1]
                try:
                    Dataset = getattr(datasets, '%s' % src)  # 执行Tension
                except:
                    raise Exception("data name type not implemented")
                self.datasets[source] = Dataset(data_path, args.normlizetype,
                                                op=op, signal_size=args.data_length).data_preprare(
                    is_src=True)  # 源域  self.datasets[source]
        for key in self.datasets.keys():
            logging.info('source set {} length {}.'.format(key, len(self.datasets[key])))
            self.datasets[key].summary()

        # 目标域数据
        if args.area == 'SDA' or args.area == 'MDA':  # SDA和MDA目标域分为训练集和验证集
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
            self.datasets['train'].summary();
            self.datasets['val'].summary()

            self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x],
                                                               batch_size=args.batch_size,
                                                               shuffle=(False if x == 'val' else True),
                                                               num_workers=args.num_workers,
                                                               drop_last=(False if x == 'val' else True),
                                                               pin_memory=(True if self.device == 'cuda' else False))
                                for x in (['train', 'val'] + args.source_name)}
            self.iters = {x: iter(self.dataloaders[x]) for x in (['train', 'val'] + args.source_name)}

        self.datasets['source_train'], self.datasets['source_val'], self.datasets['target_val'] = Dataset(args.data_dir,
                                                                    args.transfer_task,
                                                                    args.normlizetype,
                                                                    args.data_length,)\
                                                                    .data_split(transfer_learning=False)

        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_train', 'source_val']}

        # Define the model
        self.model = getattr(diagnosis_models, args.model_name)(args)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, Dataset.num_classes)

        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the optimizer
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("optimizer not implement")

        # Define the learning rate decay
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            # 阶梯衰减，每次衰减的epoch数根据列表steps给出，gamma代表学习率衰减倍数
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            # 每走一个epoch，学习率衰减args.gamma倍
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            # 每走steps个epoch，学习率衰减args.gamma倍，阶梯形式
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'cos':
            # 20代表从lr从最大到最小的epoch数，0代表学习率的最小值
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 20, 0)
        elif args.lr_scheduler == 'fix':
            # 无衰减
            self.lr_scheduler = None
        else:
            raise Exception("lr schedule not implement")

        self.start_epoch = 0

        # Invert the model and define the loss
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        return self.dataloaders['source_val']

    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()
        loss_list = []
        val_loss_list = []
        acc_list = []
        val_acc_list = []

        for epoch in range(self.start_epoch, args.max_epoch):

            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            # Update the learning rate
            if self.lr_scheduler is not None:
                # self.lr_scheduler.step(epoch)
                logging.info('current lr: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['source_train', 'source_val']:
                # Define the temp variable
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0

                if phase == 'source_train':
                    self.model.train()
                if phase == 'source_val':
                    self.model.eval()

                for batch_idx, (inputs, labels) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    # Do the learning process, in val, we do not care about the gradient for relaxing
                    with torch.set_grad_enabled(phase == 'source_train'):
                        logits = self.model(inputs)
                        loss = self.criterion(logits, labels)
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = loss.item() * inputs.size(0)
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        # Calculate the training information
                        if phase == 'source_train':
                            # backward
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += inputs.size(0)
                            # Print the training information
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(inputs), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1
                # Print the train and val information via each epoch
                epoch_loss = epoch_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = epoch_acc / len(self.dataloaders[phase].dataset)
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))
                if phase == 'source_train':
                    loss_list.append(epoch_loss)
                else:
                    val_loss_list.append(epoch_loss)

                if phase == 'source_train':
                    acc_list.append(epoch_acc)
                else:
                    val_acc_list.append(epoch_acc)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, loss_list, val_loss_list, acc_list, val_acc_list
