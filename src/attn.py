import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class TemporalAttention(nn.Module):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(TemporalAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size
        self.distances = torch.zeros((window_size, window_size))
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j)

    def forward(self, queries, keys, values, queries1, keys1, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores

        scores1 = torch.einsum("blhe,bshe->bhls", queries1, keys1)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores1.masked_fill_(attn_mask.mask, -np.inf)
        attn1 = scale * scores1

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5
        sigma = torch.pow(3, sigma) - 1
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1)
        # prior = (1.0 / (2 * sigma)) * torch.exp(-prior / sigma)  # 拉普拉斯分布
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2))  # 高斯分布
        series = self.dropout(torch.softmax(attn, dim=-1))
        self_atten = self.dropout(torch.softmax(attn1, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", self_atten, values)

        if self.output_attention:
            return V.contiguous(), series, prior, sigma
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.query_projection1 = nn.Linear(d_model,
                                           d_keys * n_heads)
        self.key_projection1 = nn.Linear(d_model,
                                         d_keys * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, queries1, keys1, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        queries1 = self.query_projection1(queries1).view(B, L, H, -1)
        keys1 = self.key_projection1(keys1).view(B, S, H, -1)
        sigma = self.sigma_projection(x).view(B, L, H)

        out, series, prior, sigma = self.inner_attention(
            queries,  # 用于series
            keys,  # 用于series
            values,
            queries1,  # 用于self
            keys1,  # 用于self
            sigma,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), series, prior, sigma


class SourceAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(SourceAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = math.sqrt(d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.layer_norm1 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, mask=None):
        batch_size = x.size(0)

        query = self.query_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key_linear(enc_output).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value_linear(enc_output).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.projection(attn_output)

        attn_output = self.dropout(attn_output)
        attn_output = self.layer_norm1(x + attn_output)

        return attn_output, attn_weights


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_size = d_model // num_heads

        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale_factor = math.sqrt(d_model)
        self.projection = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        query = self.query_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        key = self.key_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        value = self.value_linear(x).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-1, -2)) / self.scale_factor
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_size)
        attn_output = self.projection(attn_output)

        return attn_output, attn_weights
