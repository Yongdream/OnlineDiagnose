import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import ADDneck, resnet18

def mfsan_guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]

    return sum(kernel_val)

def MFSAN_mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, cb=None):

    batch_size = int(source.size()[0])
    kernels = mfsan_guassian_kernel(source, target, kernel_mul=kernel_mul,
                               kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    if cb != None:
        loss = torch.mean(XX + cb * cb.T * YY - cb * XY - cb.T * YX)
    else:
        loss = torch.mean(XX + YY - XY -YX)

    return loss

class MFSAN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, num_source=1):
        super(MFSAN, self).__init__()
        
        self.num_source = num_source
        self.sharedNet = resnet18(False, in_channel=in_channel)
        self.sonnet = nn.ModuleList([ADDneck(512, 256) for _ in range(num_source)])
        self.cls_fc_son = nn.ModuleList(([nn.Linear(256, num_classes)
                                                       for _ in range(num_source)]))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                     nn.Flatten())

    def forward(self, data_tgt, data_src=None, label_src=None, source_idx=None, device=None):
        if self.training == True:
            feat_src = self.sharedNet(data_src)
            feat_tgt = self.sharedNet(data_tgt)

            feat_tgt = [son(feat_tgt) for son in self.sonnet]
            feat_tgt = [self.avgpool(data) for data in feat_tgt]

            feat_src = self.sonnet[source_idx](feat_src)
            feat_src = self.avgpool(feat_src)
            
            loss_mmd = MFSAN_mmd(feat_src, feat_tgt[source_idx])
            
            logits_src = self.cls_fc_son[source_idx](feat_src)
            logits_tgt = [self.cls_fc_son[i](feat_tgt[i]) for i in range(self.num_source)]
            loss_cls = F.cross_entropy(logits_src, label_src)

            loss_l1 = 0.0
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
            for i in range(self.num_source - 1):
                for j in range(i+1, self.num_source):
                    loss_l1 += torch.abs(logits_tgt[i] - logits_tgt[j]).sum()               #多个分类器的分类差异
            loss_l1 /= self.num_source

            return logits_tgt[source_idx], loss_cls, loss_mmd, loss_l1
        else:
            feat = self.sharedNet(data_tgt)

            feat = [son(feat) for son in self.sonnet]
            feat = [self.avgpool(data) for data in feat]
            logits_tgt = [self.cls_fc_son[i](feat[i]) for i in range(self.num_source)]
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]
            
            pred = torch.zeros((logits_tgt[0].shape)).to(device)
            for i in range(self.num_source):
                pred += logits_tgt[i]           #分类器结果相加

            return pred
