import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps # If MSE == 0, We need eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        return loss


# Pearson Correlations Coeffiicient Loss
class PearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true) -> float:
        x = y_pred.clone()
        y = y_true.clone()
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        cov = torch.sum(vx * vy)
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        corr = torch.maximum(torch.minimum(corr, torch.tensor(1)), torch.tensor(-1))

        return torch.sub(torch.tensor(1), corr ** 2)