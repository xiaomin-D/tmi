import torch.nn.functional as F
from torch.autograd import Variable

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        return score


class MixLoss(nn.Module):
    def __init__(self, smooth=1):
        super(MixLoss, self).__init__()
        self.smooth = smooth
        # self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, logits, targets, y_pred, y_gt):
        num = targets.size(0)

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = m1 * m2

        score = (
            2.0
            * (intersection.sum(1) + self.smooth)
            / (m1.sum(1) + m2.sum(1) + self.smooth)
        )
        score = 1 - score.sum() / num
        # 将预测输出进行Softmax转换
        
        # y_pred = self.softmax(y_pred)

        #  计算交叉熵损失
        loss = -(y_gt * torch.log(y_pred) + (1 - y_gt) * torch.log(1 - y_pred)).mean()

        
        return score + loss