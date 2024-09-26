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

class train_diagnosis_SDTL_utils(object):
    def __init__(self, args, save_dir):
        self.args = args
        self.save_dir = save_dir

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

    def _get_next_batch(self, loaders, iters, device):
        try:
            inputs, labels = next(iters)
        except StopIteration:#迭代完了继续迭代
            iters = iter(loaders)
            inputs, labels = next(iters)

        return inputs.to(device), labels.to(device)

    def _get_accuracy(self, preds, targets):
        assert preds.shape[0] == targets.shape[0]
        correct = torch.eq(preds.argmax(dim=1), targets).float().sum().item()
        accuracy = correct / preds.shape[0]

        return accuracy

    def setup(self):
        """
        Initialize the datasets, model, loss and optimizer
        :param args:
        :return:
        """
        args = self.args

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


        # Load the datasets
        Dataset = getattr(datasets, args.data_name)
        self.datasets = {}

        if isinstance(args.transfer_task[0], str):
           #print(args.transfer_task)
           args.transfer_task = eval("".join(args.transfer_task))

        self.datasets['source_train'], self.datasets['target_train'], self.datasets['target_val'] = Dataset(args.data_dir,
                                                                                                            args.transfer_task,
                                                                                                            args.normlizetype,
                                                                                                            args.data_length,).data_split(transfer_learning=True)


        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           drop_last=(True if x.split('_')[
                                                               1] == 'train' else False),
                                                           pin_memory=(True if self.device == 'cuda' else False))
                            for x in ['source_train', 'target_train', 'target_val']}

        self.iters = {x: iter(self.dataloaders[x]) for x in ['source_train', 'target_train', 'target_val']}

        # Define the model
        model = getattr(diagnosis_models, args.model_name)

        if args.model_name in ['DAN', 'CDAN','DANN']:
            self.training_mode = 1
            self.model = model(in_channel=1, num_classes=Dataset.num_classes)
        else:
            raise Exception("model type not implemented")


        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)

        # Define the learning parameters
        self.optimizer = self._get_optimizer(self.model)

        # Define the optimizer
        self.lr_scheduler = self._get_lr_scheduler(self.optimizer)

        self.start_epoch = 0

        # Invert the model
        self.model.to(self.device)

        return self.dataloaders['target_val']


    def train(self):
        """
        Training process
        :return:
        """
        args = self.args

        best_acc = 0.0

        loss_list = []          #训练损失
        val_acc_list = []       #验证准确率

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch) + '-' * 5)

            # Update the learning rate
            if self.lr_scheduler is not None:
                logging.info('current lr: {}'.format(self.lr_scheduler.get_last_lr()))
            else:
                logging.info('current lr: {}'.format(args.lr))

            # Each epoch has a training and val phase
            for phase in ['target_train', 'target_val']:
                # Define the temp variable
                epoch_acc = 0.0
                epoch_loss_all = 0.0
                epoch_loss = defaultdict(float)

                # Set model to train mode or evaluate mode
                if phase == 'target_train':
                    self.model.train()
                    tradeoff = self._get_tradeoff(args.tradeoff, epoch)
                else:
                    self.model.eval()

                num_iter = len(self.iters['source_train'])    #以源域num_iter为最大iter次数

                for i in tqdm(range(num_iter), ascii=True):#开始迭代
                    #确定数据
                    if phase == 'target_train':#目标域训练阶段需要获取源域数据
                        source_data, source_labels = self._get_next_batch(self.dataloaders['source_train'],self.iters['source_train'], self.device)
                        target_data, _ = self._get_next_batch(self.dataloaders['target_train'],self.iters['target_train'], self.device)
                    else:
                        target_data, target_labels = self._get_next_batch(self.dataloaders['target_val'],self.iters['target_val'], self.device)

                    if phase == 'target_train':     #目标域训练
                        with torch.set_grad_enabled(True):
                            # forward
                            self.optimizer.zero_grad()
                            if self.training_mode == 1:  #DAN DANN CDAN
                                pred, loss_0, loss_1 = self.model(target_data,source_data, source_labels)
                                loss = loss_0 + tradeoff[0] * loss_1
                                epoch_loss[0] += loss_0; epoch_loss[1] += loss_1

                            epoch_loss_all += loss.item()

                            # backward  每一次迭代更新参数
                            loss.backward()
                            self.optimizer.step()

                    else:#目标域验证
                        with torch.no_grad():
                            pred = self.model(target_data)

                            epoch_acc += self._get_accuracy(pred, target_labels)

                # Print the train and val information via each epoch
                if phase == 'target_train':
                    for key in epoch_loss.keys():
                        logging.info('{}-Loss_{}: {:.4f}'.format(phase, key,epoch_loss[key] / num_iter))

                    loss_list.append(epoch_loss_all/ num_iter)#训练阶段损失

                else:
                    logging.info('{}-Acc: {:.4f}'.format(phase, epoch_acc))

                    val_acc_list.append(epoch_acc / num_iter)#验证阶段准确率

                    # log the best model according to the val accuracy
                    if epoch_acc >= best_acc:
                        best_acc = epoch_acc
                        best_epoch = epoch
                    logging.info("The best model epoch {}, val-acc {:.4f}".format(best_epoch,best_acc))


            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        return self.model, loss_list,val_acc_list  #返回模型,训练损失，验证准确率