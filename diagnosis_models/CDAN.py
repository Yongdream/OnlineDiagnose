import torch
from .resnet import resnet18
from torch import nn
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


class CDAN(nn.Module):

    def __init__(self, in_channel=1, num_classes=3, dropout=0):
        super(CDAN, self).__init__()

        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            resnet18(False, in_channel=in_channel),
            nn.AdaptiveAvgPool1d(4),
                        
            nn.Flatten(),
            nn.Linear(4*512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout))

        self.clf = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            
            nn.Linear(128, num_classes))

        self.discriminator = nn.Sequential(
            nn.Linear(2304, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2))

        self.grl = GradientReverseLayer()

    def forward(self, target_data, source_data=None, source_label=None):
        if self.training == True:
            batch_size = source_data.shape[0]
            feat_src = self.feature_extractor(source_data)
            feat_tgt = self.feature_extractor(target_data)
    
            logits = self.clf(feat_src)
            logits_tgt = self.clf(feat_tgt)
    
            loss = F.cross_entropy(logits, source_label)
    
            softmax_output_src = F.softmax(logits, dim=-1)
            softmax_output_tgt = F.softmax(logits_tgt, dim=-1)
           
            labels_dm = torch.concat((torch.ones(batch_size, dtype=torch.long),
                torch.zeros(batch_size, dtype=torch.long)), dim=0).to(target_data.device)
    
            feat_src_ = torch.bmm(softmax_output_src.unsqueeze(2),
                            feat_src.unsqueeze(1)).view(batch_size, self.num_classes*256)
            feat_tgt_ = torch.bmm(softmax_output_tgt.unsqueeze(2),
                            feat_tgt.unsqueeze(1)).view(batch_size, self.num_classes*256)
            feat = self.grl(torch.concat((feat_src_, feat_tgt_), dim=0))
            logits_dm = self.discriminator(feat)
            loss_dm = F.cross_entropy(logits_dm, labels_dm)
    
            return logits_tgt, loss, loss_dm
        else:
            feat_tgt = self.feature_extractor(target_data)
            logits_tgt = self.clf(feat_tgt)
            
            return logits_tgt
