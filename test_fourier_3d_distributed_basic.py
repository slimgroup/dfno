# Tests basic functionality of Distributed 3d Fourier Neural Operator

import distdl
import numpy as np
import sys
import torch

from argparse import ArgumentParser
from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from fourier_3d_distributed import DistributedFNO3d
from mpi4py import MPI

Partition = distdl.backend.backend.Partition

# Parse command line arguments
parser = ArgumentParser()
parser.add_argument('--input-shape', '-is', type=int, nargs=5, default=[20, 20, 32, 32, 32])
parser.add_argument('--input-partition-shape', '-ips', type=int, nargs=5, default=[1, 1, 2, 2, 2])
parser.add_argument('--outfile', '-o', type=str, default='fourier-3d-distributed-output.log')
parser.add_argument('--modes', type=int, nargs=3, default=[8, 8, 8])

args = parser.parse_args()

nw = np.prod(args.input_partition_shape)

# Set up network partitions
P_world = Partition(MPI.COMM_WORLD)

P_x_base = P_world.create_partition_inclusive(np.arange(nw))
P_x = P_x_base.create_cartesian_topology_partition(args.input_partition_shape)

# Create fake input data
np.random.seed(P_world.rank)
torch.manual_seed(P_world.rank)

x_global_structure = TensorStructure()
x_global_structure.shape = args.input_shape

x_local_shape = compute_subtensor_shapes_balanced(x_global_structure, P_x.shape)[tuple(P_x.index)]
x = torch.randn(*x_local_shape)

print(f'rank = {P_x.rank}, x.shape = {x.shape}')

# Set up network
network = DistributedFNO3d(P_x, args.modes[0], args.modes[1], args.modes[2], args.input_shape[1])

y = network(x)

P_x._comm.Barrier()

print(f'rank = {P_x.rank}, y.shape = {y.shape}')

dy = torch.randn(*y.shape)

y.backward(dy)
