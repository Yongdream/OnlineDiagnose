#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Wang, Xiang

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy


# function
def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def get_norm_layer(norm):
    if norm == "layer":
        return nn.LayerNorm
    elif norm == "batch":
        return nn.BatchNorm2d
    raise RuntimeError("normalization should be layer/batch, not {}".format(norm))


# sub-block
class TransformerEncoderLayer_BN(nn.Module):
    def __init__(self, d_model, n_token, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer_BN, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.BatchNorm1d(n_token)
        self.norm2 = nn.BatchNorm1d(n_token)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_BN, self).__setstate__(state)

    def forward(self, src: torch.Tensor, src_mask: [torch.Tensor] = None,
                src_key_padding_mask: [torch.Tensor] = None) -> torch.Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = src.transpose(0, 1)
        src = self.norm1(src)
        src = src.transpose(0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.transpose(0, 1)
        src = self.norm2(src)
        src = src.transpose(0, 1)
        return src


class Conv_Bn_Gelu(nn.Module):
    def __init__(self, d_model, n_token, in_ch, out_ch, k_size, dropout):
        """
        in:(N, in_ch, L, E)
        out:(N, out_ch, L, E)
        """
        super(Conv_Bn_Gelu, self).__init__()
        self.n_pad = (k_size - 1) // 2
        self.dropout = nn.Dropout(p=dropout)
        self.conv1 = nn.Conv2d(in_ch, out_ch, k_size, padding=self.n_pad)
        self.bn = nn.BatchNorm2d(out_ch)
        self.acti = nn.GELU()

    def forward(self, x):
        res = self.dropout(x)
        res = self.conv1(res)
        res = self.bn(res)
        res = self.acti(res)
        return res


class Dense_Relu(nn.Module):
    """"""
    def __init__(self, in_dim, out_dim, dropout):
        """"""
        super(Dense_Relu, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        """"""
        res = self.dropout(x)
        res = self.linear(res)
        res = self.act(res)
        return res


class ResidualBlock(nn.Module):
    """
    (N, C, L)
    """
    def __init__(self, in_ch, out_ch, k_size):
        """"""
        super(ResidualBlock, self).__init__()
        self.model_name = self.__class__.__name__
        self.conv1 = Conv_Bn_Relu(in_ch, out_ch, k_size)
        self.conv2 = Conv_Bn_Relu(out_ch, out_ch, k_size)
        self.conv3 = Conv_Bn_Relu(out_ch, out_ch, k_size)
        self.x_conv = Conv_Bn_Relu(in_ch, out_ch, 1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        res = res + self.x_conv(x)
        return res


class Conv_Bn_Relu(nn.Module):
    """"""

    def __init__(self, in_ch, out_ch, k_size):
        """
        (N, C, L)
        """
        super(Conv_Bn_Relu, self).__init__()
        padding = (k_size - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, k_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.conv(x)
        res = self.bn(res)
        res = self.relu(res)
        return res


# define Embedding(Tokenize)
class Embedding_Conv(nn.Module):
    """
    in:(N, L, in_dim)
    out:(N, L, out_dim)
    """
    def __init__(self, n_token, k_size, dropout):
        super(Embedding_Conv, self).__init__()
        n_pad = (k_size - 1) // 2
        self.dropout = nn.Dropout(p=dropout)
        self.conv = nn.Conv1d(n_token, n_token, k_size, padding=n_pad)

    def forward(self, x):
        res = self.dropout(x)
        res = self.conv(res)
        return res


# define Position Encoding
class PositionalEncoding_Fixed(nn.Module):
    """"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """"""
        super(PositionalEncoding_Fixed, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        if d_model > 2:
            pe[:, 0::2] = torch.sin(position * div_term)  # [:, 0::2]，[axis=0所有的数据，axis=2从0开始取值，间隔为2]
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.sin(position * div_term)  # [:, 0::2]，[axis=0所有的数据，axis=2从0开始取值，间隔为2]
            # pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """"""
        temp_pe = self.pe[:, 0:x.shape[1], 0:x.shape[2]]
        x = (x * math.sqrt(self.d_model)) + temp_pe
        return self.dropout(x)


class PositionalEncoding_Learnable(nn.Module):
    """"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """"""
        super(PositionalEncoding_Learnable, self).__init__()
        self.d_model = d_model
        self.learn_pe = nn.Parameter(torch.randn(1, max_len, d_model))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        temp_pe = self.learn_pe[:, 0:x.shape[1], 0:x.shape[2]]
        x = (x * math.sqrt(self.d_model)) + temp_pe
        return x


# define Encoder
class Encoder_TransformerEncoder_LN(nn.Module):
    """"""
    def __init__(self, d_model, nhd, nly, dropout, hid):
        """"""
        super(Encoder_TransformerEncoder_LN, self).__init__()
        encoder = nn.TransformerEncoderLayer(d_model, nhead=nhd, dim_feedforward=hid,
                                             dropout=dropout, activation='gelu')
        self.encoder_lays = nn.TransformerEncoder(encoder, nly, norm=None)

    def forward(self, X):
        """
        in:(N, L, E)
        """
        res = X.transpose(0, 1)
        # (L, N, E)
        res = self.encoder_lays(res)
        res = res.transpose(0, 1)
        return res


class Encoder_TransformerEncoder_BN(nn.Module):
    """"""
    def __init__(self, d_model, n_token, nhd, nly, dropout, hid):
        """"""
        super(Encoder_TransformerEncoder_BN, self).__init__()
        encoder = TransformerEncoderLayer_BN(d_model, n_token, nhead=nhd, dim_feedforward=hid,
                                             dropout=dropout, activation='gelu')
        self.encoder_lays = nn.TransformerEncoder(encoder, nly, norm=None)

    def forward(self, X):
        """
        in:(N, L, E)
        """
        res = X.transpose(0, 1)
        # (L, N, E)
        res = self.encoder_lays(res)
        res = res.transpose(0, 1)
        return res


class Decoder_Linear(nn.Module):
    """
    in:(N, L, E)
    """
    def __init__(self, d_model: int, n_token: int, dropout: float):
        super(Decoder_Linear, self).__init__()
        self.flt = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))
        self.linear = nn.Linear(n_token * d_model, 2)
        # self.activ = nn.Sigmoid()
        self.activ = nn.Softmax(dim=1)

    def forward(self, x):
        res = self.flt(x)
        res = self.linear(res)
        res = self.activ(res)
        return res


class Decoder_Linear_CLS(nn.Module):
    """
    in:(N, L, E)
    """
    def __init__(self, d_model: int, dropout: float):
        super(Decoder_Linear_CLS, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_model, 2)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        res = x[:, 0, :]
        res = self.dropout(res)
        res = self.linear(res)
        res = self.sigm(res)
        return res


# define Decoder
class Decoder_MLP_Linear(nn.Module):
    """"""
    def __init__(self, d_model, n_token, dropout=0.1):
        """"""
        super(Decoder_MLP_Linear, self).__init__()
        linear = nn.Linear
        activ = nn.Sigmoid
        self.flt = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))
        self.hidden1 = nn.Sequential(linear(d_model * n_token, 256), activ())
        self.dropout2 = nn.Dropout(p=dropout)
        # self.hidden2 = nn.Sequential(linear(d_model // 2, d_model // 4), activ())
        # self.dropout3 = nn.Dropout(p=dropout)
        self.linear = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        """"""
        res = self.flt(X)
        res = self.hidden1(res)
        res = self.dropout2(res)
        # res = self.hidden2(res)
        # res = self.dropout3(res)
        res = self.linear(res)
        res = self.softmax(res)
        return res


class Decoder_Dense(nn.Module):
    """"""
    def __init__(self, d_model: int, n_token: int, dropout: float = 0.1):
        """"""
        super(Decoder_Dense, self).__init__()
        self.flt = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))
        self.linear = nn.Linear(n_token * d_model, 2)
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        res = self.flt(x)
        res = self.sigm(self.linear(res))
        return res


class Decoder_Conv_Bn_Gelu(nn.Module):
    """"""
    def __init__(self, d_model, n_token, dropout):
        """
        in:(N, L, E)
        out:(N, d_cls)
        """
        super(Decoder_Conv_Bn_Gelu, self).__init__()
        self.d_model = d_model
        self.n_token = n_token
        # conv in
        self.conv1 = Conv_Bn_Gelu(d_model, n_token, 1, 192, 7, dropout)

        # inception
        self.incp1 = Conv_Bn_Gelu(d_model, n_token, 192, 64, 1, dropout)
        self.incp2 = nn.Sequential(Conv_Bn_Gelu(d_model, n_token, 192, 96, 1, dropout),
                                   Conv_Bn_Gelu(d_model, n_token, 96, 128, 3, dropout))
        self.incp3 = nn.Sequential(Conv_Bn_Gelu(d_model, n_token, 192, 16, 1, dropout),
                                   Conv_Bn_Gelu(d_model, n_token, 16, 32, 5, dropout))
        self.incp4 = nn.Sequential(nn.MaxPool2d(3, 1, (3-1)//2),
                                   Conv_Bn_Gelu(d_model, n_token, 192, 32, 1, dropout))

        # our layer
        self.linear = nn.Linear(256, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.conv1(x)
        res_1 = self.incp1(res)
        res_2 = self.incp2(res)
        res_3 = self.incp3(res)
        res_4 = self.incp4(res)
        res = torch.cat([res_1, res_2, res_3, res_4], dim=1)
        # global pooling
        res = F.adaptive_avg_pool2d(res, 1)
        res = res.squeeze().squeeze()
        # out
        res = self.linear(res)
        res = self.softmax(res)
        return res


class Conv_Norm_Activ(nn.Module):
    """
    in:(N, in_ch, L, E)
    our:(N, out_ch, L, E)
    """
    def __init__(self, d_model, in_ch, out_ch, k_size: tuple, dropout, activation='relu', norm='layer'):
        """"""
        super(Conv_Norm_Activ, self).__init__()
        if norm == 'layer':
            self.norm_layer = get_norm_layer(norm)(d_model)
        elif norm == 'batch':
            self.norm_layer = get_norm_layer(norm)(out_ch)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.activation_layer = get_activation_fn(activation)

        self.n_pad = ((k_size[0] - 1) // 2, (k_size[1] - 1) // 2)
        self.conv_layer = nn.Conv2d(in_ch, out_ch, k_size, padding=self.n_pad)


    def forward(self, x):
        """"""
        res = self.dropout_layer(x)
        res = self.conv_layer(res)
        res = self.norm_layer(res)
        res = self.activation_layer(res)
        return res


class Decoder_Inception(nn.Module):
    """
    in:(N, L, E)
    """
    def __init__(self, d_model, n_token, n_head, dropout, activation='relu', norm='batch'):
        super(Decoder_Inception, self).__init__()
        self.head_dim = d_model // n_head
        self.loc_ksz = (5, 5)
        self.F_tok = 3
        self.F_dim = 2 * self.head_dim
        if self.F_dim % 2 == 0:
            self.F_dim += 1

        self.F_n_tok = n_token
        if self.F_n_tok % 2 == 0:
            self.F_n_tok += 1

        self.F_d_model = d_model
        if self.F_d_model % 2 == 0:
            self.F_d_model += 1

        self.conv = Conv_Norm_Activ(d_model, 1, 32, (1, 1), dropout, activation=activation, norm=norm)

        # self.conv1 = Conv_Norm_Activ(d_model, 32, 16, (1, 1), dropout, activation=activation, norm=norm)

        self.cov_loc = Conv_Norm_Activ(d_model, 32, 16, (3, 3), dropout, activation=activation, norm=norm)

        self.pool_conv1 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=(3 - 1) // 2),
                                        Conv_Norm_Activ(d_model, 32, 16, (1, 1), dropout, activation=activation,
                                                        norm=norm))

        # our layer
        self.linear = nn.Linear(32, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x_ = x.unsqueeze(dim=1)
        res = self.conv(x_)
        # res_1 = self.conv1(res)
        res_2 = self.pool_conv1(res)
        res_3 = self.cov_loc(res)
        res = torch.cat([res_2, res_3], dim=1)
        # global pooling
        res = F.adaptive_avg_pool2d(res, 1)
        res = res.squeeze().squeeze()
        res = self.linear(res)
        res = self.softmax(res)
        return res


# define Model
class PE_fixed_EC_transformer_DC_mlp_linear(nn.Module):
    """"""

    def __init__(self, d_model, flatten_len, nhd=8, nly=6, dropout=0.1, hid=2048):
        """"""
        self.model_name = self.__class__.__name__
        super(PE_fixed_EC_transformer_DC_mlp_linear, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)
        # encoder
        self.encoder = Encoder_TransformerEncoder_LN(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)
        # flatten
        self.flt = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))
        # decoder
        self.decoder = Decoder_MLP_Linear(flatten_len, dropout)

    def forward(self, X):
        """"""
        pass


class PE_fixed_EC_transformer_DC_dense(nn.Module):
    """"""
    def __init__(self, d_model: int, n_token: int,
                 nhd: int = 8, nly: int = 6, dropout: float = 0.1, hid: int = 2048):
        """"""
        self.model_name = self.__class__.__name__
        super(PE_fixed_EC_transformer_DC_dense, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)
        # encoder
        self.encoder = Encoder_TransformerEncoder_LN(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)
        # decoder
        self.decoder = Decoder_Dense(d_model, n_token, dropout)

    def forward(self, X):
        """"""
        res = self.position_encoding(X)
        res = self.encoder(res)
        res = self.decoder(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


class PE_fixed_EC_transformer_DC_conv_pooling(nn.Module):
    """"""
    def __init__(self, d_model: int, n_token: int, nhd: int = 8, nly: int = 6, dropout: float = 0.1, hid: int = 2048):
        """"""
        self.model_name = self.__class__.__name__
        super(PE_fixed_EC_transformer_DC_conv_pooling, self).__init__()
        # position encoding
        self.position_encoding = PositionalEncoding_Fixed(d_model, dropout=dropout)
        # encoder
        self.encoder = Encoder_TransformerEncoder_LN(d_model, nhd=nhd, nly=nly, dropout=dropout, hid=hid)
        # decoder
        #self.decoder = nn.Sequential(nn.Dropout(p=dropout), Decoder_Conv_Pooling(d_model, n_token, nhd))
        self.flt = nn.Sequential(nn.Flatten(), nn.Dropout(p=dropout))
        self.linear = nn.Linear(n_token * d_model, 2)
        self.sigm = nn.Sigmoid()

    def forward(self, X):
        """"""
        res = self.position_encoding(X)
        res = self.encoder(res)
        #res = self.decoder(res)
        res = self.flt(res)
        res = self.linear(res)
        res = self.sigm(res)
        return res


class PureMLP(nn.Module):
    """"""
    def __init__(self, in_dim: int = None, dropout: float = 0.1) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(PureMLP, self).__init__()
        pass

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = 0
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class EB_cov_PE_lr_EC_trans_DC_conv(nn.Module):
    """
    in:(N, L, in_dim)
    out:(N, d_cls)
    """
    def __init__(self,
                 t_len: int,
                 d_model: int,
                 n_token: int,
                 nhd: int = 8,
                 nly: int = 6,
                 dropout: float = 0.1,
                 hid: int = 2048) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(EB_cov_PE_lr_EC_trans_DC_conv, self).__init__()
        self.eb_conv = Embedding_Conv(t_len, d_model, n_token, 3, dropout)
        self.pe_lr = PositionalEncoding_Learnable(d_model, dropout=dropout)
        self.encoder = Encoder_TransformerEncoder_BN(d_model, n_token, nhd, nly, dropout, hid)
        self.decoder = Decoder_Conv_Bn_Gelu(d_model, n_token, dropout)

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = self.eb_conv(x)
        res = self.pe_lr(res)
        res = self.encoder(res)
        res = self.decoder(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


class BaseLine_MLP(nn.Module):
    """"""
    def __init__(self, in_dim):
        """"""
        super(BaseLine_MLP, self).__init__()
        self.model_name = self.__class__.__name__
        # dense
        self.dense1 = Dense_Relu(in_dim, 500, 0.1)
        self.dense2 = Dense_Relu(500, 500, 0.2)
        self.dense3 = Dense_Relu(500, 500, 0.2)

        # last layer
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(500, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """"""
        res = self.dense1(x)
        res = self.dense2(res)
        res = self.dense3(res)
        res = self.dropout(res)
        res = self.linear(res)
        res = self.softmax(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_FCN(nn.Module):
    """"""
    def __init__(self, in_dim):
        """"""
        super(BaseLine_FCN, self).__init__()
        self.model_name = self.__class__.__name__
        # conv layer
        self.conv1 = Conv_Bn_Relu(1, 128, 7)
        self.conv2 = Conv_Bn_Relu(128, 256, 5)
        self.conv3 = Conv_Bn_Relu(256, 128, 3)

        # last layer
        self.linear = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.conv1(x)
        res = self.conv2(res)
        res = self.conv3(res)
        # global pooling
        res = F.adaptive_avg_pool1d(res, 1)
        res = res.squeeze()
        # out layer
        res = self.linear(res)
        res = self.softmax(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class BaseLine_ResNet(nn.Module):
    """"""
    def __init__(self, in_dim):
        """"""
        super(BaseLine_ResNet, self).__init__()
        self.model_name = self.__class__.__name__
        self.residual1 = ResidualBlock(1, 64, 7)
        self.residual2 = ResidualBlock(64, 128, 5)
        self.residual3 = ResidualBlock(128, 128, 3)
        # out layer
        self.linear = nn.Linear(128, 2)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """"""
        x = x.unsqueeze(dim=1)
        res = self.residual1(x)
        res = self.residual2(res)
        res = self.residual3(res)
        # global pooling
        res = F.adaptive_avg_pool1d(res, 1)
        res = res.squeeze()
        # out layer
        res = self.linear(res)
        res = self.softmax(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        return model


class MyModel(nn.Module):
    """
    in:(N, L, in_dim)
    out:(N, d_cls)
    """
    def __init__(self,
                 d_model: int,
                 n_token: int,
                 nhd: int = 8,
                 nly: int = 6,
                 dropout: float = 0.1,
                 hid: int = 2048) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(MyModel, self).__init__()
        self.embedding = Embedding_Conv(n_token, 3, dropout)
        self.position_encode = PositionalEncoding_Learnable(d_model, dropout=dropout)
        self.encoder = Encoder_TransformerEncoder_LN(d_model, nhd, nly, dropout, hid)
        self.decoder = Decoder_Inception(d_model, n_token, nhd, dropout, activation='gelu', norm='layer')
        # self.decoder = Decoder_Linear(d_model, n_token, dropout)

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = self.embedding(x)
        res = self.position_encode(res)
        res = self.encoder(res)
        res = self.decoder(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


class MyModel_DC_Linear(nn.Module):
    """
    in:(N, L, in_dim)
    out:(N, d_cls)
    """
    def __init__(self,
                 d_model: int,
                 n_token: int,
                 nhd: int = 8,
                 nly: int = 6,
                 dropout: float = 0.1,
                 hid: int = 2048) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(MyModel_DC_Linear, self).__init__()
        self.embedding = Embedding_Conv(n_token, 3, dropout)
        self.position_encode = PositionalEncoding_Learnable(d_model, dropout=dropout)
        self.encoder = Encoder_TransformerEncoder_LN(d_model, nhd, nly, dropout, hid)
        self.decoder = Decoder_Linear(d_model, n_token, dropout)

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = self.embedding(x)
        res = self.position_encode(res)
        res = self.encoder(res)
        res = self.decoder(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


class MyModel_DC_MLP(nn.Module):
    """
    in:(N, L, in_dim)
    out:(N, d_cls)
    """
    def __init__(self,
                 d_model: int,
                 n_token: int,
                 nhd: int = 8,
                 nly: int = 6,
                 dropout: float = 0.1,
                 hid: int = 2048) -> None:
        """"""
        self.model_name = self.__class__.__name__
        super(MyModel_DC_MLP, self).__init__()
        self.embedding = Embedding_Conv(n_token, 3, dropout)
        self.position_encode = PositionalEncoding_Learnable(d_model, dropout=dropout)
        self.encoder = Encoder_TransformerEncoder_LN(d_model, nhd, nly, dropout, hid)
        self.decoder = Decoder_MLP_Linear(d_model, n_token, dropout)

    def forward(self, x: torch.tensor) -> None:
        """"""
        res = self.embedding(x)
        res = self.position_encode(res)
        res = self.encoder(res)
        res = self.decoder(res)
        return res

    @classmethod
    def init_model(cls, init_dic: dict):
        """"""
        model = cls(**init_dic)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model


if __name__ == '__main__':
    pass
