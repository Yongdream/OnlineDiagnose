import torch
import logging
import torch.nn as nn
from .resnet import resnet18
import torch.nn.functional as F
from torch.autograd import Function

class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx, input, coeff = 1.):
        ctx.coeff = coeff
        output = input * 1.0

        return output

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output.neg() * ctx.coeff, None


class GradientReverseLayer(nn.Module):
    def __init__(self):
        super(GradientReverseLayer, self).__init__()

    def forward(self, *input):
        return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):

    def __init__(self, alpha = 1.0, lo = 0.0, hi = 1.,
                      max_iters = 1000., auto_step = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input):
        coeff = np.float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo)
        if self.auto_step:
            self.step()

        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        self.iter_num += 1

def binary_accuracy(output, target):

    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


class DomainAdversarialLoss(nn.Module):

    def __init__(self, domain_discriminator, reduction = 'mean', grl = None):
        super(DomainAdversarialLoss, self).__init__()
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=1.,
                                 max_iters=1000, auto_step=True) if grl is None else grl
        self.domain_discriminator = domain_discriminator
        self.bce = lambda input, target, weight: \
            F.binary_cross_entropy(input, target, weight=weight, reduction=reduction)

    def forward(self, f_s, f_t, w_s = None, w_t = None):
        f = self.grl(torch.cat((f_s, f_t), dim=0))
        d = self.domain_discriminator(f)
        d_s, d_t = d.chunk(2, dim=0)
        d_label_s = torch.ones((f_s.size(0), 1)).to(f_s.device)
        d_label_t = torch.zeros((f_t.size(0), 1)).to(f_t.device)
        
        
        d_accuracy = 0.5 * (binary_accuracy(d_s, d_label_s) \
                                            + binary_accuracy(d_t, d_label_t))

        if w_s is None:
            w_s = torch.ones_like(d_label_s)
        if w_t is None:
            w_t = torch.ones_like(d_label_t)
        loss = 0.5 * (self.bce(d_s, d_label_s, w_s.view_as(d_s)) + \
                                   self.bce(d_t, d_label_t, w_t.view_as(d_t)))
        return loss, d_accuracy


class DomainDiscriminator(nn.Sequential):

    def __init__(self, in_feature, hidden_size, batch_norm = True):
        if batch_norm:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )
        else:
            super(DomainDiscriminator, self).__init__(
                nn.Linear(in_feature, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(hidden_size, 1),
                nn.Sigmoid()
            )

    def get_parameters(self):

        return [{"params": self.parameters(), "lr": 1.}]


class DANN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3):
        super(DANN, self).__init__()
        self.classifier = nn.Sequential(
                resnet18(False, in_channel=in_channel),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU())
        self.head =  nn.Linear(256, num_classes)

        domain_discri = DomainDiscriminator(in_feature=256, hidden_size=512)
        grl = GradientReverseLayer()
        self.domain_adv = DomainAdversarialLoss(domain_discri, grl=grl)

    def forward(self, data_tgt, data_src=None, label_src=None):
        if self.training == True:
            x = torch.cat((data_src, data_tgt), dim=0)
    
            f = self.classifier(x)
            y = self.head(f)
            f_s, f_t = f.chunk(2, dim=0)
            y_s, y_t = y.chunk(2, dim=0)
    
            loss_cls = F.cross_entropy(y_s, label_src)
            loss_d, acc_d = self.domain_adv(f_s, f_t)
    
            return y_t, loss_cls, loss_d
        else:
            f = self.classifier(data_tgt)
            pred = self.head(f)
            
            return pred
