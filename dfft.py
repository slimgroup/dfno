import distdl
import numpy as np
import torch.nn as nn
import torch

from distdl.utilities.torch import *
from torch import Tensor
from typing import Any, Callable, List, NewType, Optional
from utils import *

Partition = distdl.backend.backend.Partition
Transform = NewType('Transform', Callable[[Tensor], Tensor])

def create_dxfftn_decomposition_partition(P_x: Partition, dims: List[int], mul_dims: Optional[List[int]] = None) -> Partition:
    '''Creates a partition over which sequential transforms will be performed
    on every dimension of size 1.

    Arguments
    ---------

    P_x : Partition
        The base partition
    
    dims : List[int]
        The dimensions which will have size 0

    mul_dims : Optional[List[int]]
        List of dimensions to multiply dims into to keep the number of workers
        the same when repartitioning. Defaults to the next set of dimensions of
        the same size above dims, modulo the shape of the partition without
        overlap if none provided.

    Returns
    -------

    P_decomposition : Partition
        A cartesian partition with the same number of workers as P_x, and size
        1 on the given dims.

    '''

    n = P_x.dim
    k = len(dims)
    
    if mul_dims is None:
        mul_dims = [(d + k) % n for d in dims]
        mul_dims = [n-1 if m in dims else m for m in mul_dims]

    P_shape = np.copy(P_x.shape)

    for d, m in zip(dims, mul_dims):
        P_shape[d] = 1
        P_shape[m] = P_shape[m]*P_x.shape[d]

    return P_x.create_cartesian_topology_partition(P_shape)

class DXFFTN(nn.Module):
    '''Performs a distributed n-dimensional fast fourier transform operator from
    the given list of transforms on a tensor distributed over the given
    partition.

    Arguments
    ---------

    P_x : Partition
        The partition over which the input tensor is distributed

    dims : Optional[List[int]]
        A list of dimensions over which to perform the transforms. Defaults to
        all dimensions if none are provided.

    transforms : Optional[List[Transform]]
        A list of transforms to apply on a per-dimension basis. These need not
        be fourier transforms, just any function which is differentiable, is a
        linear operator, and has a tensor as input and output. Defaults to
        performing an n-dimensional fast fourier transform on every given
        dimension if none are provided.

    decomposition_order : Optional[int]
        The number of dimensions with partition size 1 in each fft partition.
        E.g. in 3d, a pencil decomposition has decomposition_order = 1, while
        a slab decomposition has decomposition_order = 2, etc.
    '''
    

    def __init__(self,
                 P_x: Partition,
                 dims: Optional[List[int]] = None,
                 transforms: Optional[List[Transform]] = None,
                 decomposition_order: Optional[int] = 1) -> None:

        super(DXFFTN, self).__init__()

        self.P_x = P_x
        self.dims = np.arange(P_x.dim) if dims is None else dims
        self.transforms = [torch.fft.fftn for _ in range(len(self.dims))] if transforms is None else transforms
        self.decomposition_order = decomposition_order
        
        self.partition_dims = []
        for i in range(0, len(self.dims), self.decomposition_order):
            j = min(len(self.dims), i + self.decomposition_order)
            self.partition_dims.append(self.dims[i:j])

        self.partitions = [create_dxfftn_decomposition_partition(self.P_x, ds) for ds in self.partition_dims]
        self.transposes = [distdl.nn.DistributedTranspose(P_i, P_j) for [P_i, P_j] in window_iter(self.partitions, 2)]
        self.transposes.append(distdl.nn.DistributedTranspose(self.partitions[-1], self.P_x))
        self.transpose_in = distdl.nn.DistributedTranspose(self.P_x, self.partitions[0])

    def forward(self, x: Tensor) -> Tensor:
        
        x = self.transpose_in(x)

        for f, T, ds in zip(self.transforms, self.transposes, self.partition_dims):
            x = f(x, dim=tuple(ds))
            x = T(x)

        return x


if __name__ == '__main__':
    
    import matplotlib.pyplot as plt

    from argparse import ArgumentParser
    from matplotlib import cm
    from mpi4py import MPI
        
    # Parse program arguments
    parser = ArgumentParser()

    parser.add_argument('--partition-shape', '-ps', type=int, nargs='+')
    parser.add_argument('--sampling-rate', '-sr', type=int, default=32)
    parser.add_argument('--decomposition-order', '-do', type=int, default=1)
    parser.add_argument('--display', '-d', action='store_true')
    parser.add_argument('--verbose', '-v', action='store_true')

    args = parser.parse_args()
    
    # Create partitions
    nw = np.prod(args.partition_shape)
    n = len(args.partition_shape)

    P_world = Partition(MPI.COMM_WORLD)
    
    P_0_base = P_world.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1] * n)

    P_x_base = P_world.create_partition_inclusive(np.arange(nw))
    P_x = P_x_base.create_cartesian_topology_partition(args.partition_shape)

    if args.verbose and P_0.active:
        print(f'==== Partitions ====')
        print(f'P_world.size = {P_world.size}')
        print(f'P_x.shape = {P_x.shape}')

    # Setup distribution layers
    scatter = distdl.nn.DistributedTranspose(P_0, P_x)
    gather = distdl.nn.DistributedTranspose(P_x, P_0)

    # Setup FFT inputs
    sr = args.sampling_rate
    dt = 1/sr

    if P_0.active:
        
        axes = [np.linspace(0, 1, sr) for _ in range(n)]
        Xs = np.meshgrid(*axes)
        
        Y0 = np.zeros_like(Xs[0])
        for i, X in enumerate(Xs):
            Y0 += np.sin(2*np.pi*(i+1)*X)

        Y0 = Tensor(Y0)
        Y1 = Y0.clone()
    
        if args.display and n == 2:
            fig = plt.figure(figsize=(18, 6))
            ax = fig.add_subplot(131, projection='3d')
            ax.plot_surface(*Xs, Y0.numpy(), cmap=cm.jet)
            ax.set_title('Signal')

    else:
        Y1 = zero_volume_tensor()


    # Perform sequential and distributed FFTs
    if P_0.active:
        F0 = torch.fft.fftn(Y0)

    dfftn = DXFFTN(P_x, decomposition_order=args.decomposition_order)

    if args.verbose and P_0.active:
        print(f'==== FFT Partitions ====')
        print('dims -> partition shape')
        for ds, P in zip(dfftn.partition_dims, dfftn.partitions):
            print(f'{ds} -> {P.shape}')

    Y1 = scatter(Y1)
    F1 = dfftn(Y1)
    F1 = gather(F1)

    # Show error and plot results
    if P_0.active:
       
        err = np.linalg.norm(F1-F0)
        print(f'norm(F1-F0) = {err}')

        if args.display and n == 2:
            ax = fig.add_subplot(132, projection='3d')
            ax.plot_surface(*Xs, F0.numpy().imag, cmap=cm.jet)
            ax.set_title('FFTN (Torch)')

            ax = fig.add_subplot(133, projection='3d')
            ax.plot_surface(*Xs, F1.numpy().imag, cmap=cm.jet)
            ax.set_title('DFFTN (Custom)')
            
            plt.show()
