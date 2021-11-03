import distdl
import distdl.nn as dnn
from distdl.utilities.torch import zero_volume_tensor
import numpy as np
import torch
import torch.nn as nn

from distdl.functional import ZeroVolumeCorrectorFunction

Partition = distdl.backend.backend.Partition

def create_root_partition(P):
    P_0_base = P.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1]*len(P.shape))
    return P_0

def create_standard_partitions(shape):
    from mpi4py import MPI
    P_world = Partition(MPI.COMM_WORLD)
    P_x_base = P_world.create_partition_inclusive(np.arange(np.prod(shape)))
    P_x = P_x_base.create_cartesian_topology_partition(shape)
    P_0 = create_root_partition(P_x)
    return P_world, P_x, P_0

def unit_guassian_normalize(x):
    mu = torch.mean(x, 0).unsqueeze(0)
    std = torch.std(x, 0).unsqueeze(0)
    out = (x-mu)/(std+1e-6)
    return out, mu, std

def unit_gaussian_denormalize(x, mu, std):
    return x*(std + 1e-6) + mu

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

def generate_batch_indices(P_x, n, bs, shuffle=False):
    P_0 = create_root_partition(P_x)
    if P_0.active:
        indices = []
        for i in range(0, n, bs):
            start = i
            stop = min(i+bs, n)
            indices.append((start, stop))

        if shuffle:
            np.random.shuffle(indices)

        indices = torch.tensor(indices)
    
    else:
        indices = zero_volume_tensor()

    B = dnn.Broadcast(P_0, P_x)
    indices = B(indices)
    return indices

def grid_like(x):
    grids = [torch.tensor(np.linspace(0, 1, n), dtype=x.dtype) for n in x.shape[2:]]
    grids_out = []
    for i, grid in enumerate(grids):
        shape = [1]*x.dim()
        shape[i+2] = x.shape[i+2]
        repeat = list(x.shape)
        repeat[i+2] = 1
        grids_out.append(grid.reshape(shape).repeat(repeat))
    return torch.cat(tuple(grids_out), dim=1).to(x.device)