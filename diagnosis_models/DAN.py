import logging

from torch import nn
import torch.nn.functional as F
import torch

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


class FeatureExtractor(nn.Module):

    def __init__(self, in_channel=1):
        super(FeatureExtractor, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 32, kernel_size=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True))

        self.layer5 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        return x


class DAN(nn.Module):

    def __init__(self, in_channel=2, num_classes=3):
        super(DAN, self).__init__()

        self.fs = FeatureExtractor(in_channel=in_channel)

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 4, 256),
            nn.ReLU(inplace=True))

        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True))

        self.fc3 = nn.Linear(64, num_classes)

        self.mkmmd = MultipleKernelMaximumMeanDiscrepancy(
            kernels=[GaussianKernel(alpha=2 ** k) for k in range(-3, 2)])

    def forward(self, target_data, source_data=None, source_label=None):
        if self.training == True:
            f_s = self.fs(source_data)
            f_t = self.fs(target_data)

    
            f_s = self.fc1(f_s)
            f_t = self.fc1(f_t)

            loss_mmd = self.mkmmd(f_s, f_t)
            
            f_s = self.fc2(f_s)
            f_t = self.fc2(f_t)
            loss_mmd += self.mkmmd(f_s, f_t)
    
            f_s = self.fc3(f_s)
            pred = self.fc3(f_t)
            loss_mmd += self.mkmmd(f_s, pred)
            loss = F.cross_entropy(f_s, source_label)
            return pred, loss, loss_mmd
        else:
            f_t = self.fs(target_data)
            f_t = self.fc1(f_t)
            f_t = self.fc2(f_t)
            pred = self.fc3(f_t)
            return pred