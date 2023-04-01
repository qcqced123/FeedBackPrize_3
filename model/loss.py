import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # If MSE == 0, We need eps

    def forward(self, yhat, y) -> Tensor:
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


class MCRMSELoss(nn.Module):
    # num_scored => setting your number of metrics
    def __init__(self, num_scored=6):
        super().__init__()
        self.rmse = RMSELoss()
        self.num_scored = num_scored

    def forward(self, yhat, y):
        score = 0
        for i in range(self.num_scored):
            score += self.rmse(yhat[:, i], y[:, i]) / self.num_scored
        return score


# Pearson Correlations Co-efficient Loss
class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(self, y_pred, y_true) -> Tensor:
        x = y_pred.clone()
        y = y_true.clone()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr = torch.maximum(torch.minimum(corr, torch.tensor(1)), torch.tensor(-1))
        return torch.sub(torch.tensor(1), corr ** 2)


# Cross-Entropy Loss
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_pred, y_true) -> Tensor:
        criterion = nn.CrossEntropyLoss()
        return criterion(y_pred, y_true)


# Binary Cross-Entropy Loss
class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_pred, y_true) -> Tensor:
        criterion = nn.BCEWithLogitsLoss()
        return criterion(y_pred, y_true)
