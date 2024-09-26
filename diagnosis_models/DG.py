import logging

import utils
import torch
from torch import nn
import torch.nn.functional as F

class MMSD(nn.Module):
    def __init__(self):
        super(MMSD, self).__init__()

    def _mix_rbf_mmsd(self, X, Y, sigmas=(1,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmsd(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX, dim1=-2, dim2=-1)
        Y_sqnorms = torch.diagonal(YY, dim1=-2, dim2=-1)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmsd(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        m = torch.tensor(K_XX.size(0), dtype=torch.float32)
        n = torch.tensor(K_YY.size(0), dtype=torch.float32)
        C_K_XX = torch.pow(K_XX, 2)
        C_K_YY = torch.pow(K_YY, 2)
        C_K_XY = torch.pow(K_XY, 2)
        if biased:
            mmsd = (torch.sum(C_K_XX) / (m * m) + torch.sum(C_K_YY) / (n * n)
            - 2 * torch.sum(C_K_XY) / (m * n))
        else:
            if const_diagonal is not False:
                trace_X = m * const_diagonal
                trace_Y = n * const_diagonal
            else:
                trace_X = torch.trace(C_K_XX)
                trace_Y = torch.trace(C_K_YY)

            mmsd = ((torch.sum(C_K_XX) - trace_X) / ((m - 1) * m)
                    + (torch.sum(C_K_YY) - trace_Y) / ((n - 1) * n)
                    - 2 * torch.sum(C_K_XY) / (m * n))
        return mmsd

    def forward(self, X1, X2, bandwidths=[3]):
        kernel_loss = self._mix_rbf_mmsd(X1, X2, sigmas=bandwidths)
        return kernel_loss

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


class DG(nn.Module):

    def __init__(self, in_channel=1, num_classes=2, num_source=2):
        super(DG, self).__init__()

        self.num_classes = num_classes
        self.num_source = num_source
        self.shared_fs = SharedFeatureExtractor(1, 128)

        self.specific_fs = nn.ModuleList(SpecificFeatureExtractor(128, 128) \
                                              for _ in range(num_source))

        self.clf = nn.ModuleList(SpecificClassifier(128, num_classes) \
                                              for _ in range(num_source))

        self.mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])
        self.mmsd=MMSD()

    def coral(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff
          
    def forward(self,data_tgt=None, device=None, source_data=[], source_label=[]):
        if self.training:  # 训练阶段需要计算源域分类损失
            assert len(source_data) == len(source_label) == self.num_source
            
            shared_feat = [self.shared_fs(data) for data in source_data]
            specific_feat = [self.specific_fs[i](shared_feat[i]) for i in range(self.num_source)]

            logits = [self.clf[i](specific_feat[i]) for i in range(self.num_source)]
            loss_cls = 0.0
            for i in range(self.num_source):
                loss_cls += F.cross_entropy(logits[i], source_label[i])#源域分类损失
                # logging.info('{}{}'.format(logits[i], source_label[i]))
                #logging.info('{}'.format(loss_cls))
        
            loss_coral = 0.0
            for i in range(self.num_source - 1):
                for k in range(i + 1, self.num_source):  # （0，1）（0，2）（1，2)
                    #coral
                    #loss_coral+=self.coral(specific_feat[i], specific_feat[k])

                    #mmsd
                    #loss_coral += self.mmsd(specific_feat[i], specific_feat[k])
                    #logging.info('{}'.format(loss_coral))

                    #mmd
                    loss_coral += self.mkmmd(specific_feat[i], specific_feat[k])

            return loss_cls, loss_coral
        else:
            #logging.info('{}'.format(data_tgt.shape))
            shared_feat_tgt = self.shared_fs(data_tgt)
            specific_feat_tgt = [fs(shared_feat_tgt) for fs in self.specific_fs]

            logits_tgt = [self.clf[i](specific_feat_tgt[i]) for i in range(self.num_source)]
            logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt]

            pred = torch.zeros((logits_tgt[0].shape)).to(torch.device("cuda"))
            for i in range(self.num_source):
                pred += logits_tgt[i]       #分类器结果相加
            pred=pred/self.num_source
            #logging.info('{}'.format(pred))

            return pred
