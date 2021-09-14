import distdl
import numpy as np
import torch

from collections import OrderedDict
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import *
from mpi4py import MPI
from torch import Tensor

from adjoint_test import check_adjoint_test_tight
from spectral_conv import DistributedSpectralConvNd

Partition = distdl.backend.backend.Partition

P_world = Partition(MPI.COMM_WORLD)
P_0_base = P_world.create_partition_inclusive([0])
P_0 = P_0_base.create_cartesian_topology_partition([1, 1, 1, 1, 1, 1])
P_x_base = P_world.create_partition_inclusive(np.arange(16))
P_x = P_x_base.create_cartesian_topology_partition([1, 1, 2, 2, 2, 2])

x_global_shape = [8, 10, 16, 16, 16, 16]

layer = DistributedSpectralConvNd(P_x, in_channels=x_global_shape[1], out_channels=x_global_shape[1], modes=[5, 6, 7, 8])

x = zero_volume_tensor(x_global_shape[0])
if P_x.active:
    x_local_shape = compute_subshape(P_x.shape, P_x.index, x_global_shape)
    x = torch.randn(*x_local_shape)

x.requires_grad = True

y = layer(x)
dy = zero_volume_tensor(x_global_shape[0], dtype=y.dtype)
if layer.P_y.active:
    dy = torch.randn(*y.shape, dtype=y.dtype)

y.backward(dy)
dx = x.grad

x = x.detach()
dx = dx.detach()
dy = dy.detach()
y = y.detach()

check_adjoint_test_tight(P_world, x, dx, y, dy)