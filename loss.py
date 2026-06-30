import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def total_loss(pred, target):

    obj_loss = F.binary_cross_entropy_with_logits(
        pred[..., 4],
        target[..., 4]
    )

    box_loss = F.mse_loss(
        pred[..., :4],
        target[..., :4]
    )

    cls_loss = F.binary_cross_entropy_with_logits(
        pred[..., 5:],
        target[..., 5:]
    )

    return obj_loss + box_loss + cls_loss


class YoloLoss(nn.Module):
    def __init__(self, S, B, C):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        pass
