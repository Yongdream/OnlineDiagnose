#!/usr/bin/python
# -*- coding:utf-8 -*-

from diagnosis_models.ANN import ANN_Net as ANN
from diagnosis_models.CNN import CNN_1D_Net as CNN
from diagnosis_models.LSTM import LSTM_Net as LSTM
from diagnosis_models.GRU import GRU_Net as GRU
from diagnosis_models.Bi_LSTM import Bi_LSTM_Net as Bi_LSTM
from diagnosis_models.DRSN import DRSN_Net as DRSN
from diagnosis_models.DRN import DRN_Net as DRN

from diagnosis_models.DAN import DAN
from diagnosis_models.DANN import DANN
from diagnosis_models.CDAN import CDAN

from diagnosis_models.MFSAN import MFSAN
from diagnosis_models.MSSA import MSSA
from diagnosis_models.MINE import MINE

from diagnosis_models.DG import DG