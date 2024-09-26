import utils
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Any, Tuple

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


class CORAL(nn.Module):
    def __init__(self):
        super(CORAL, self).__init__()

    def forward(self, source, target):
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.mul((xc - xct), (xc - xct))
        loss = torch.sum(loss) / (4 * d * d)
        return loss

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class MinimumClassConfusionLoss(nn.Module):
    def __init__(self, temperature: float):
        super(MinimumClassConfusionLoss, self).__init__()
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_classes = logits.shape
        predictions = F.softmax(logits / self.temperature, dim=1)  # batch_size x num_classes
        entropy_weight = Entropy(predictions).detach()
        entropy_weight = 1 + torch.exp(-entropy_weight)
        entropy_weight = (batch_size * entropy_weight / torch.sum(entropy_weight)).unsqueeze(dim=1)  # batch_size x 1

        class_confusion_matrix = torch.mm((predictions * entropy_weight).transpose(1, 0), predictions) # num_classes x num_classes
        class_confusion_matrix = class_confusion_matrix / torch.sum(class_confusion_matrix, dim=1)
        mcc_loss = (torch.sum(class_confusion_matrix) - torch.trace(class_confusion_matrix)) / num_classes
        return mcc_loss


class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class SpecificClassifier(nn.Module):  ###每一对源域和目标域的分类器

    def __init__(self, in_channel, num_classes):
        super(SpecificClassifier, self).__init__()

        self.clf = nn.Sequential(
            nn.Linear(in_channel, 64),
            #nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout(),

            nn.Linear(64, num_classes))

    def forward(self, input):
        y = self.clf(input)

        return y


class SharedFeatureExtractor(nn.Module):  ##共享的特征提取器

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


class SpecificFeatureExtractor(nn.Module):  ##每一对源域和目标域的特定特征提取器

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
            nn.ReLU(inplace=True))
        self.avgpool = nn.AdaptiveAvgPool1d(4)
        self.maxpool = nn.AdaptiveMaxPool1d(4)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * in_channel, out_channel),
            nn.ReLU(inplace=True))

    def forward(self, input):
        feat = self.feature_extractor(input)
        feat_pool = self.avgpool(feat) + self.maxpool(feat)
        #feat_pool = self.maxpool(feat)
        feat_fc = self.fc(feat_pool)

        return feat_fc


class MINE(nn.Module):

    def __init__(self, in_channel=1, num_classes=2, num_source=1):
        super(MINE, self).__init__()

        self.num_classes = num_classes
        self.num_source = num_source
        self.shared_fs = SharedFeatureExtractor(1, 128)

        self.specific_fs = nn.ModuleList(SpecificFeatureExtractor(128, 128) \
                                         for _ in range(num_source))

        self.clf = nn.ModuleList(SpecificClassifier(128, num_classes) \
                                 for _ in range(num_source))

        #loss
        #self.MMD_loss = MMDLoss()
        self.coral_loss = CORAL()
        self.mcc_loss_s = MinimumClassConfusionLoss(temperature=2.5)  # 类别混淆损失
        self.mcc_loss_t = MinimumClassConfusionLoss(temperature=2.5)#类别混淆损失

        self.mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

        self.mmsd = MMSD()


    def forward(self, target_data, device=None, source_data=[], source_label=[]):



        shared_feat_tgt = self.shared_fs(target_data)
        specific_feat_tgt = [fs(shared_feat_tgt) for fs in self.specific_fs]

        logits_tgt_ = [self.clf[i](specific_feat_tgt[i]) for i in range(self.num_source)]
        logits_tgt = [F.softmax(data, dim=1) for data in logits_tgt_]

        

        if self.training:  # 训练阶段需要计算分类损失
            assert len(source_data) == len(source_label) == self.num_source  # source_data是由两个tensor组成的列表
            shared_feat = [self.shared_fs(data) for data in source_data]
            specific_feat = [self.specific_fs[i](shared_feat[i]) for i in range(self.num_source)]

            logits = [self.clf[i](specific_feat[i]) for i in range(self.num_source)]

            loss_cls_s = 0.0
            loss_cls_t = 0.0
            for i in range(self.num_source):
                loss_cls_s += nn.CrossEntropyLoss()(logits[i], source_label[i])  # 源域分类损失
                #loss_cls_s += self.mcc_loss_s(logits[i]) # 源域mcc损失
                
            # for i in range(self.num_source):
            #     loss_cls_t += self.mcc_loss_t(logits_tgt_[i])  # 目标域mcc损失
                
            loss_lmmd = []  # lmmd
            loss_mmd = []  # mmd
            for i in range(self.num_source):
                lmmd_single_src = 0.0

                #coral
                # lmmd_single_src = self.coral_loss(specific_feat[i], specific_feat_tgt[i])
                # loss_lmmd.append(lmmd_single_src)

                #mmd
                # lmmd_single_src = self.mkmmd(specific_feat[i], specific_feat_tgt[i])
                # loss_lmmd.append(lmmd_single_src)

                # mmsd
                lmmd_single_src = self.mmsd(specific_feat[i], specific_feat_tgt[i])
                loss_lmmd.append(lmmd_single_src)


                #lmmd
                # oh_label = one_hot(source_label[i], self.num_classes)
                # for j in range(self.num_classes):
                #     w_src = oh_label[:, j].view(-1, 1).to(device)
                #     w_tgt = logits_tgt[i][:, j].view(-1, 1).to(device)
                #     lmmd_single_src += self.mkmmd(w_src * specific_feat[i], w_tgt * specific_feat_tgt[i])
                # loss_lmmd.append(lmmd_single_src / (self.num_classes))
            sum_lmmd = sum(loss_lmmd)

            for i in range(self.num_source-1):
                for j in range(i+1, self.num_source):
                    mmd_single_src = 0.0

                    #coral
                    # mmd_single_src = self.coral_loss(nn.AdaptiveAvgPool1d(1)(shared_feat[i]).view(shared_feat[i].size(0), -1), nn.AdaptiveAvgPool1d(1)(shared_feat[j]).view(shared_feat[j].size(0), -1))
                    # loss_mmd.append(mmd_single_src)

                    #mmd
                    # mmd_single_src = self.mkmmd(nn.AdaptiveAvgPool1d(1)(shared_feat[i]).view(shared_feat[i].size(0), -1),
                    #     nn.AdaptiveAvgPool1d(1)(shared_feat[j]).view(shared_feat[j].size(0), -1))
                    # loss_mmd.append(mmd_single_src)

                    # mmsd
                    mmd_single_src = self.mmsd(nn.AdaptiveAvgPool1d(1)(shared_feat[i]).view(shared_feat[i].size(0), -1),
                        nn.AdaptiveAvgPool1d(1)(shared_feat[j]).view(shared_feat[j].size(0), -1))
                    loss_mmd.append(mmd_single_src)

                    #lmmd
                    # oh_labeli = one_hot(source_label[i], self.num_classes)
                    # oh_labelj = one_hot(source_label[j], self.num_classes)
                    # for k in range(self.num_classes):
                    #     w_srci = oh_labeli[:, k].view(-1, 1).to(device)
                    #     w_srcj = oh_labelj[:, k].view(-1, 1).to(device)
                    #     mmd_single_src += self.mkmmd(w_srci * nn.AdaptiveAvgPool1d(1)(shared_feat[i]).view(shared_feat[i].size(0), -1), w_srcj * nn.AdaptiveAvgPool1d(1)(shared_feat[j]).view(shared_feat[j].size(0), -1))
                    # loss_mmd.append(mmd_single_src / (self.num_classes))

            sum_mmd = sum(loss_mmd)

            

             

        pred = torch.zeros((logits_tgt[0].shape)).to(device)
        for i in range(self.num_source):
            pred += logits_tgt[i]

        if self.training:
            return pred, (loss_cls_s+loss_cls_t), (sum_lmmd+sum_mmd)
        else:
            return pred
