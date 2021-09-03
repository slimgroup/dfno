import distdl
import numpy as np
import torch.nn as nn
import torch

from distdl.utilities.torch import *
from torch import Tensor
from typing import Any, Callable, Dict, List, NewType, Optional
from utils import *

Partition = distdl.backend.backend.Partition
Transform = NewType('Transform', Callable[[Tensor], Tensor])
PartitionDimsMapping = NewType('PartitionDimsMapping', Callable[[List[int]], Partition])

def default_partition_dims_mapping(P_x: Partition, dims_all: List[List[int]], dims: List[int]) -> Partition:
    
    l = len(dims_all)
    
    # Get the index of the passed dimension list in the list of all
    # dimension lists
    index = 0
    for i, ds in enumerate(dims_all):
        if np.array_equal(ds, dims):
            index = i
    
    # By default, take the dims we will multiply into to be the next
    # list of dims in the list of lists
    next_index = (index + 1) % l
    next_dims = dims_all[next_index]
    k = len(next_dims)

    mul_dims = [next_dims[d%k] for d in dims]
    
    # Assemble partition shape by setting the indices of the passed dims
    # to 1 and the computed mul dims to the product of dims and mul dims
    P_shape = np.copy(P_x.shape)
    for d, m in zip(dims, mul_dims):
        P_shape[d] = 1
        P_shape[m] *= P_x.shape[d]

    return P_x.create_cartesian_topology_partition(P_shape)

class DXFFTN(nn.Module):
    '''Performs a distributed n-dimensional fast fourier transform operator from
    the given list of transforms on a tensor distributed over the given
    partition.

    Arguments
    ---------

    P_x : Partition
        The partition over which the input tensor is distributed

    P_y : Optional[Partition]
        The partition over which the output tensor should be distributed.
        Defaults to the partition of the last performed transformation if
        none is provided.

    dims : Optional[List[int]]
        A list of dimensions over which to perform the transforms. Defaults to
        all dimensions if none are provided. The transforms are applied to the
        last dimension first, to the first dimension.

    partition_dims_mapping : Optional[PartitionDimsMapping]
        Mapping from each list of transform partition dims to the shape of the
        corresponding transform partition. Defaults to a modulo operator over
        dims if none is provided.

    transforms : Optional[List[Transform]]
        A list of transforms to apply on a per-dimension basis. These need not
        be fourier transforms, just any function which is differentiable, is a
        linear operator, and has a tensor as input and output. Defaults to
        performing an n-dimensional fast fourier transform on every given
        dimension if none are provided.

    transform_kwargs : Optional[List[Dict]]
        List of keyword arguments to apply on a per-transform basis. Defaults to
        a list of dictionaries with kwargs to fftn if none are specified.

    decomposition_order : Optional[int]
        The number of dimensions with partition size 1 in each fft partition.
        E.g. in 3d, a pencil decomposition has decomposition_order = 1, while
        a slab decomposition has decomposition_order = 2, etc.
    '''

    def __init__(self,
                 P_x: Partition,
                 P_y: Optional[Partition] = None,
                 dims: Optional[List[int]] = None,
                 partition_dims_mapping: Optional[PartitionDimsMapping] = None,
                 transforms: Optional[List[Transform]] = None,
                 transform_kwargs: Optional[List[Dict]] = None,
                 decomposition_order: Optional[int] = 1) -> None:

        super(DXFFTN, self).__init__()

        self.P_x = P_x
        self.n = P_x.dim

        self.dims = np.arange(self.n) if dims is None else dims
        self.d = len(self.dims)
        
        self.decomposition_order = decomposition_order
        
        # Partition dimensions are of size decomposition_order, except in the
        # last dimension if decomposition_order does not divide P_x.dim
        self.partition_dims = []
        for i in range(0, self.d, self.decomposition_order):
            j = min(i + self.decomposition_order, self.d)
            self.partition_dims.append(self.dims[i:j])
        
        # Use the default dims mapping if none is provided
        self.partition_dims_mapping = lambda ds: default_partition_dims_mapping(self.P_x, self.partition_dims, ds) \
                if partition_dims_mapping is None else partition_dims_mapping

        self.partitions = [self.partition_dims_mapping(ds) for ds in self.partition_dims]
        
        # Default to fftn if no transforms are provided
        self.transforms = [torch.fft.fftn for _ in range(len(self.partitions))] if transforms is None \
                else transforms

        self.transform_kwargs = [{'dim': tuple(ds)} for ds in self.partition_dims] if transform_kwargs is None else transform_kwargs

        # We compute the transforms in the reverse order to which they are passed
        # to mimic the behavior of fftw and torch
        self.partition_dims = list(reversed(self.partition_dims))
        self.partitions = list(reversed(self.partitions))
        self.transforms = list(reversed(self.transforms))
        self.transform_kwargs = list(reversed(self.transform_kwargs))

        # Set up transpose operators
        self.P_y = self.partitions[-1] if P_y is None else P_y
        self.transposes = [distdl.nn.DistributedTranspose(P_i, P_j) for [P_i, P_j] in window_iter(self.partitions, 2)]
        self.transposes.append(distdl.nn.DistributedTranspose(self.partitions[-1], self.P_y))
        self.transpose_in = distdl.nn.DistributedTranspose(self.P_x, self.partitions[0])

    def forward(self, x: Tensor) -> Tensor:

        x = self.transpose_in(x)

        for f, kwargs, T, ds in zip(self.transforms, self.transform_kwargs, self.transposes, self.partition_dims):
            x = f(x, **kwargs)
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
        F0 = torch.fft.rfftn(Y0)

    drfftn = DXFFTN(P_x, P_y=P_x, transforms=[torch.fft.fftn, torch.fft.rfftn], decomposition_order=args.decomposition_order)

    if args.verbose and P_0.active:
        print(f'==== FFT Partitions ====')
        print('dims -> fft partition shape')
        for ds, P in zip(drfftn.partition_dims, drfftn.partitions):
            print(f'{ds} -> {P.shape}')

    Y1 = scatter(Y1)
    F1 = dfftn(Y1)
    F1 = gather(F1)

    # Show error and plot results
    if P_0.active:
       
        err = np.linalg.norm(F1-F0)
        print(f'norm(F1-F0) = {err}')

        if args.display and n == 2:
            
            print(F0.shape)
            print(F1.shape)

            Xs_fft = np.meshgrid(*[np.linspace(0, 1, a) for a in reversed(F0.shape)])

            ax = fig.add_subplot(132, projection='3d')
            ax.plot_surface(*Xs_fft, F0.numpy().imag, cmap=cm.jet)
            ax.set_title('FFTN (Torch)')

            ax = fig.add_subplot(133, projection='3d')
            ax.plot_surface(*Xs_fft, F1.numpy().imag, cmap=cm.jet)
            ax.set_title('DFFTN (Custom)')
            
            plt.show()
