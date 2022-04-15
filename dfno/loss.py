import distdl.nn as dnn
import torch.nn as nn
import torch

from distdl.functional import ZeroVolumeCorrectorFunction
from .utils import create_root_partition

class DistributedRelativeLpLoss(nn.Module):

    def __init__(self, P_x, p=2):
        super(DistributedRelativeLpLoss, self).__init__()
        
        self.P_x = P_x
        self.p = p
        
        self.P_0 = create_root_partition(P_x)
        self.sr0 = dnn.SumReduce(P_x, self.P_0)
        self.sr1 = dnn.SumReduce(P_x, self.P_0)

    def forward(self, y_hat, y):
        batch_size = y_hat.shape[0]
        y_hat_flat = y_hat.reshape(batch_size, -1)
        y_flat = y.reshape(batch_size, -1)

        num = torch.sum(torch.pow(torch.abs(y_hat_flat-y_flat), self.p), dim=1)
        denom = torch.sum(torch.pow(torch.abs(y_flat), self.p), dim=1)
        num_global = self.sr0(num)
        denom_global = self.sr1(denom)

        if self.P_0.active:
            num_global = torch.pow(num_global, 1/self.p)
            denom_global = torch.pow(denom_global, 1/self.p)

        out = torch.mean(num_global/denom_global)
        return ZeroVolumeCorrectorFunction.apply(out)
