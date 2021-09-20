import distdl
import numpy as np
import torch
import torch.nn as nn

from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import *
from torch import Tensor
from typing import List, Optional

Partition = distdl.backend.backend.Partition

def check_gradient(P: Partition,
                   F: nn.Module,
                   input_shape: List[int],
                   max_iter: int = 5,
                   dtype: torch.dtype = torch.float32) -> None:

    P_0_base = P.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1]*P.dim)

    S = distdl.nn.DistributedTranspose(P_0, P)
    G = distdl.nn.DistributedTranspose(P, P_0)
    B = distdl.nn.Broadcast(P_0, P)

    criterion = distdl.nn.DistributedMSELoss(P)

    x0 = zero_volume_tensor(input_shape[0], dtype=dtype)
    x = zero_volume_tensor(input_shape[0], dtype=dtype)

    if P_0.active:
        x = torch.randn(input_shape)
        x0 = x + (1**(-P.dim))*torch.randn(input_shape)

    x = S(x)
    x0 = S(x0)
    dx = x - x0

    x.requires_grad = True
    x0.requires_grad = True

    y = F(x)
    y0 = F(x0)
    dy = y - y0
    y.backward(dy)

    f0 = criterion(y, y0).item()

    x_grad = x.grad.detach().numpy()
    x = x.detach().numpy()
    x0 = x0.detach().numpy()
    dx = dx.detach().numpy()

    h = zero_volume_tensor()
    if P_0.active:
        h = Tensor([1e-3 * f0])
        err1 = np.zeros(shape=max_iter)
        err2 = np.zeros(shape=max_iter)

    h = B(h)
    h = float(h[0])

    for j in range(max_iter):

        x_new = Tensor(x0 + h*dx)
        y_new = F(x_new)
        f = criterion(y, y_new).item()

        if P_0.active:
            err1[j] = abs(f - f0)
            err2[j] = abs(f - f0 + h*np.inner(dx.flatten(), x_grad.flatten()))
            
            print(f'{err1[j]}; {err2[j]}')

            h = h/(2*f0)
            h = Tensor([h])

        else:
            h = zero_volume_tensor()

        h = B(h)
        h = float(h[0])

    if P_0.active:
        assert np.isclose(err1[-1] / (err1[0]/2**(max_iter-1)), f0, atol=f0)
        assert np.isclose(err2[-1] / (err1[0]/4**(max_iter-1)), f0, atol=f0)
    else:
        assert True