import distdl
import numpy as np
import torch

from adjoint_test import check_adjoint_test_tight
from collections import OrderedDict
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import *
from mpi4py import MPI
from torch import Tensor

from dxfftn import DXFFTN

Partition = distdl.backend.backend.Partition

P_world = Partition(MPI.COMM_WORLD)
P_0_base = P_world.create_partition_inclusive([0])
P_0 = P_0_base.create_cartesian_topology_partition([1, 1, 1, 1, 1, 1])
P_x_base = P_world.create_partition_inclusive(np.arange(16))
P_x = P_x_base.create_cartesian_topology_partition([1, 1, 2, 2, 2, 2])

x_global_shape = [8, 10, 23, 27, 33, 45]

info = OrderedDict([
	((4, 5), {
		'transform': torch.fft.rfftn,
		'transform_kwargs': {'dim': (4, 5)},
		'repartition_dims': (2, 3)
	}),
	((2, 3), {
		'transform': torch.fft.fftn,
		'transform_kwargs': {'dim': (2, 3)},
		'repartition_dims': (4, 5)
	})
])

layer = DXFFTN(P_x, info, P_y=P_x)

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