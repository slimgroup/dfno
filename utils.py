import distdl.nn as dnn
import numpy as np
import torch

from distdl.utilities.torch import zero_volume_tensor

def unit_gaussian_normalize(x, eps=1e-6):
    mu = torch.mean(x, 0).unsqueeze(0)
    std = torch.std(x, 0).unsqueeze(0)
    out = (x-mu)/(std+eps)
    return out, mu, std, eps

def generate_batch_indices(P_0, P_x, num_data, batch_size, shuffle=False):
    if P_0.active:
        batch_indices = np.array([[i, min(i+batch_size, num_data)] for i in range(0, num_data, batch_size)])
        if shuffle:
            np.random.shuffle(batch_indices)
        batch_indices = torch.tensor(batch_indices)
    else:
        batch_indices = zero_volume_tensor()

    B = dnn.Broadcast(P_0, P_x)
    batch_indices = B(batch_indices)
    return batch_indices

def grid_like(x):

    device = x.device
    dim = len(x.shape)
    dim_grids = []

    for i, d in enumerate(x.shape[2:]):
        j = i+2
        grid = torch.tensor(np.linspace(0, 1, d), dtype=torch.float)
        shape = [1]*dim
        shape[j] = d
        repeat = list(x.shape)
        repeat[1] = 1
        repeat[j] = 1
        grid = grid.reshape(*shape).repeat(tuple(repeat))
        dim_grids.append(grid)

    return torch.cat(tuple(dim_grids), dim=1).to(device)

def count_params(parameters):
    n = 0
    for p in parameters:
        n += np.prod(p.size() + (2,) if p.is_complex() else p.size())
    return n
