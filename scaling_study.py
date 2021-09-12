import distdl
import numpy as np
import torch

from argparse import ArgumentParser
from collections import OrderedDict
from distdl.utilities.slicing import compute_subshape
from distdl.utilities.torch import *
from mpi4py import MPI
from timer import Timer
from torch import Tensor

from fno_nd import DistributedFNONd

Partition = distdl.backend.backend.Partition

parser = ArgumentParser()
parser.add_argument('--input-shape',           '-is',  type=int, nargs='+',         help='Global input shape (b x c x d_n x ... x d_0')
parser.add_argument('--input-partition-shape', '-ips', type=int, nargs='+',         help='Input partition shape (p_b x p_c x p_d_n x ... p_d_0)')
parser.add_argument('--modes',                 '-m',   type=int, nargs='+',         help='Number of modes in each data dimension')
parser.add_argument('--width',                 '-w',   type=int, default=20,        help='Channel size of lifted tensor')
parser.add_argument('--num-batches',           '-nb',  type=int, default=10,        help='Number of batches to run')
parser.add_argument('--output',                '-o',   type=str, default='out.txt', help='Data output file')

args = parser.parse_args()

assert len(args.input_shape) == len(args.input_partition_shape) == len(args.modes)+2
assert args.input_partition_shape[0] == args.input_partition_shape[1] == 1

dim = len(args.input_shape)
nw = np.prod(args.input_partition_shape)
x_global_shape = args.input_shape
y_global_shape = np.copy(args.input_shape)
y_global_shape[1] = 1

P_world = Partition(MPI.COMM_WORLD)
P_0_base = P_world.create_partition_inclusive([0])
P_0 = P_0_base.create_cartesian_topology_partition([1] * dim)
P_x_base = P_world.create_partition_inclusive(np.arange(nw))
P_x = P_x_base.create_cartesian_topology_partition(args.input_partition_shape)

with open(args.output, 'w') as f:
    f.write(f'{np.array(x_global_shape)}\n')
    f.write(f'{P_x.shape}\n')

torch.manual_seed(P_world.rank)

network = DistributedFNONd(P_x, modes=args.modes, width=args.width)

parameters = [p for p in network.parameters()]
if not parameters:
    parameters = [nn.Parameter(torch.zeros(1))]

criterion = distdl.nn.DistributedMSELoss(P_x)
optimizer = torch.optim.Adam(parameters, lr=1e-3)

timer = Timer(P_x)

for i in range(args.num_batches):

    timer.start('batch')

    timer.start('data')

    x = zero_volume_tensor(x_global_shape[0])
    y = zero_volume_tensor(x_global_shape[0])

    if P_x.active:

        x_local_shape = compute_subshape(P_x.shape, P_x.index, x_global_shape)
        y_local_shape = compute_subshape(P_x.shape, P_x.index, y_global_shape)

        x = torch.randn(*x_local_shape)
        y = torch.randn(*y_local_shape)

    x.requires_grad = True

    timer.stop('data', f'{i}')

    optimizer.zero_grad()

    timer.start('forward')
    out = network(x)
    timer.stop('forward', f'{i}')

    timer.start('loss')
    loss = criterion(out, y)
    loss_value = loss.item()
    timer.stop('loss', f'{i}')

    if P_0.active:
        print(f'batch = {i}, loss = {loss_value}')

    timer.start('adjoint')
    loss.backward()
    timer.stop('adjoint', f'{i}')

    timer.start('step')
    optimizer.step()
    timer.stop('step', f'{i}')

    timer.stop('batch', f'{i}')

    with open(args.output, 'a') as f:
        timer.dump_times(f)