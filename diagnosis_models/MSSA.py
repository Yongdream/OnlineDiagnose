import torch
from torch import nn
import torch.nn.functional as F
import logging

def one_hot(x, class_count):
    return torch.eye(class_count)[x,:]

def _update_index_matrix(batch_size, index_matrix = None, linear = True):
    if index_matrix is None or index_matrix.size(0) != batch_size * 2:
        index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        if linear:
            for i in range(batch_size):
                s1, s2 = i, (i + 1) % batch_size
                t1, t2 = s1 + batch_size, s2 + batch_size
                index_matrix[s1, s2] = 1. / float(batch_size)
                index_matrix[t1, t2] = 1. / float(batch_size)
                index_matrix[s1, t2] = -1. / float(batch_size)
                index_matrix[s2, t1] = -1. / float(batch_size)
        else:
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j:
                        index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
                        index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
            for i in range(batch_size):
                for j in range(batch_size):
                    index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
                    index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)

    return index_matrix


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):

    def __init__(self, kernels, linear = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = kernels
        self.index_matrix = None
        self.linear = linear

    def forward(self, z_s, z_t):
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0))
        self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

        # Add up the matrix of each kernel
        kernel_matrix = sum([kernel(features) for kernel in self.kernels])
        # Add 2 / (n-1) to make up for the value on the diagonal
        # to ensure loss is positive in the non-linear version
        loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)

        return loss


class GaussianKernel(nn.Module):

    def __init__(self, sigma = None, track_running_stats = True, alpha = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X):
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))


class SpecificClassifier(nn.Module):###每一对源域和目标域的分类器

    def __init__(self, in_channel, num_classes):
        super(SpecificClassifier, self).__init__()

        self.clf = nn.Sequential(
             nn.Linear(in_channel, 64),
             nn.ReLU(inplace=True),

             nn.Linear(64, num_classes))

    def forward(self, input):
        y = self.clf(input)

        return y


class SharedFeatureExtractor(nn.Module):##共享的特征提取器

    def __init__(self, in_channel=1, out_channel=128):
        super(SharedFeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True))

    def forward(self, input):
        feat = self.feature_extractor(input)

        return feat


class SpecificFeatureExtractor(nn.Module):##每一对源域和目标域的特定特征提取器

    def __init__(self, in_channel=128, out_channel=128):
        super(SpecificFeatureExtractor, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),

            nn.Conv1d(in_channel, in_channel, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_channel),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),

            nn.Flatten(),
            nn.Linear(4*in_channel, out_channel),
            nn.ReLU(inplace=True))

    def forward(self, input):
        feat = self.feature_extractor(input)

        return feat


class MSSA(nn.Module):

    def __init__(self, in_channel=1, num_classes=2, num_source=1):
        super(MSSA, self).__init__()

        self.num_classes = num_classes
        self.num_source = num_source
        self.shared_fs = SharedFeatureExtractor(1, 128)

        self.specific_fs = nn.ModuleList(SpecificFeatureExtractor(128, 128) \
                                              for _ in range(num_source))

        self.clf = nn.ModuleList(SpecificClassifier(128, num_classes) \
                                              for _ in range(num_source))

        self.mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
          
    def forward(self, target_data, device=None,source_data=[], source_label=[]):

        # 训练阶段输入源域、源域标签、目标域训练集
        # 验证阶段出入目标域验证集
        shared_feat_tgt = self.shared_fs(target_data)
        specific_feat_tgt = [fs(shared_feat_tgt) for fs in self.specific_fs]
        
        logits_tgt = [self.clf[i](specific_feat_tgt[i]) for i in range(self.num_source)]
        logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]



        if self.training:#训练阶段需要计算分类损失、特征差异损失
            assert len(source_data) == len(source_label) == self.num_source

            shared_feat = [self.shared_fs(data) for data in source_data]
            specific_feat = [self.specific_fs[i](shared_feat[i]) for i in range(self.num_source)]

            logits = [self.clf[i](specific_feat[i]) for i in range(self.num_source)]
            loss_cls = 0.0
            for i in range(self.num_source):
                loss_cls += F.cross_entropy(logits[i], source_label[i])#源域分类损失
        
            loss_mmd = []
            #mkmmd
            # for i in range(self.num_source):
            #     mmd_single_src = self.mkmmd(specific_feat[i], specific_feat_tgt[i])
            #     loss_mmd.append(mmd_single_src)
            #lmmd
            for i in range(self.num_source):
                mmd_single_src = 0.0
                oh_label = one_hot(source_label[i], self.num_classes)
                for j in range(self.num_classes):
                    w_src = oh_label[:, j].view(-1, 1).to(device)
                    w_tgt = logits_tgt[i][:, j].view(-1, 1).to(device)
                    mmd_single_src += self.mkmmd(w_src*specific_feat[i], w_tgt*specific_feat_tgt[i])
                loss_mmd.append(mmd_single_src/self.num_classes)
            sum_mmd = sum(loss_mmd)
        
        pred = torch.zeros_like(logits_tgt[0]).to(device)

        #用权重加权分类器（验证阶段）
        # for i in range(self.num_source):
        # pred += loss_mmd[i] / sum_mmd * logits_tgt[i]

        for i in range(self.num_source):
            pred += logits_tgt[i]
        
        if self.training:
            #logging.info("{}, {}, {}".format(pred,loss_cls.item(), sum_mmd.item()))
            return pred, loss_cls, sum_mmd
        else:
            return pred
