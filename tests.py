import numpy as np
import torch

from adjoint_test import check_adjoint_test_tight
from dfno import *
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import *
from gradcheck import check_gradient
from mpi4py import MPI

def test_dxfftn(P_world):

    P_world._comm.Barrier()
    
    P_0_base = P_world.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1, 1, 1, 1, 1, 1])
    P_x_base = P_world.create_partition_inclusive(np.arange(16))
    P_x = P_x_base.create_cartesian_topology_partition([1, 1, 2, 2, 2, 2])

    if P_0.active:
        print('==== Test DXFFTN ====')

    x_global_shape = [8, 10, 23, 27, 33, 45]

    info = OrderedDict([
        ((4, 5), {
            'transform': torch.fft.rfftn,
            'transform_kwargs': {'dim': (4, 5)},
            'prod_dims': (2, 3)
        }),
        ((2, 3), {
            'transform': torch.fft.fftn,
            'transform_kwargs': {'dim': (2, 3)},
            'prod_dims': (4, 5)
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

def test_spectral_conv(P_world):

    P_world._comm.Barrier()

    P_0_base = P_world.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1, 1, 1, 1, 1, 1])
    P_x_base = P_world.create_partition_inclusive(np.arange(16))
    P_x = P_x_base.create_cartesian_topology_partition([1, 1, 2, 2, 2, 2])

    if P_0.active:
        print('==== Test Spectral Conv ====')

    x_global_shape = [1, 20, 16, 16, 16, 40]
    width = x_global_shape[1]
    modes = [5, 6, 7, 8]

    layer = DistributedSpectralConvNd(P_x, width, width, modes, P_y=P_x)

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

    check_adjoint_test_tight(P_x, x, dx, y, dy)

def test_fno(P_world):

    P_0_base = P_world.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1, 1, 1, 1, 1])
    P_x_base = P_world.create_partition_inclusive(np.arange(16))
    P_x = P_x_base.create_cartesian_topology_partition([1, 1, 2, 2, 2, 2])

    if P_0.active:
        print('==== Test FNO ====')

    x_global_shape = [1, 4, 16, 16, 16, 10]
    width = x_global_shape[1]
    modes = [5, 6, 7, 8]
    out_timesteps = 40

    layer = DistributedFNONd(P_x, width, modes, out_timesteps)

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

    check_adjoint_test_tight(P_x, x, dx, y, dy)
    check_gradient(P_x, layer, x_global_shape)

if __name__ == '__main__':
    P_world = distdl.backend.backend.Partition(MPI.COMM_WORLD)
    test_dxfftn(P_world)
    test_spectral_conv(P_world)
    test_fno(P_world)
