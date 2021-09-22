import distdl
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn

from argparse import ArgumentParser
from distdl.utilities.torch import *
from fno_nd import DistributedFNONd
from mat73 import loadmat
from mpi4py import MPI
from pathlib import Path
from scipy import io
from timer import Timer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional

# Shorthand for convenience and typing
Partition = distdl.backend.backend.Partition

def create_grid(global_shape: List[int]) -> Tensor:
    '''
    Creates a grid of (0..1..d_1) x ... x (0..1..d_n) repeated along the batch dimension
    '''
    
    batch_size = global_shape[0]
    grid_shape = global_shape[2:]

    d = len(grid_shape)
    grid_subsections = []
    for i, s in enumerate(grid_shape):
        reshape = [1]*(d+2)
        reshape[i+2] = s
        repeat = [batch_size, 1, *grid_shape]
        repeat[i+2] = 1

        reshape = tuple(reshape)
        repeat = tuple(repeat)

        grid_s = Tensor(np.linspace(0, 1, s)) \
                .reshape(reshape) \
                .repeat(repeat)

        grid_subsections.append(grid_s)

    return torch.cat(tuple(grid_subsections), dim=1)

def unit_gaussian_encode(P_x: Partition, P_0: Partition, x: Tensor, eps=1e-6) -> Tensor:
    sr = distdl.nn.SumReduce(P_x, P_0)
    bc = distdl.nn.Broadcast(P_0, P_x)
    mu = torch.mean(x, 0)
    mu = sr(mu)
    mu = bc(mu)
    mu = mu / P_x.size
    P_x._comm.Barrier()
    s = torch.std(x, 0)
    s = sr(s)
    s = bc(s)
    s = s / P_x.size
    P_x._comm.Barrier()
    out = (x-mu)/(s+eps)
    return x, mu, s

def unit_gaussian_decode(x: Tensor, mu, sigma, eps=1e-6) -> Tensor:
    out = x*(sigma+eps) + mu
    return out

parser = ArgumentParser()

# Data configuration
parser.add_argument('--batch-size', '-bs', type=int, default=10, help='Batch size of training input')
parser.add_argument('--input', '-i', type=Path, help='Path to .mat file')
parser.add_argument('--input-steps', '-is', type=int, default=10, help='Number of solution steps to use as input')
parser.add_argument('--subsampling-rate', '-sr', type=int, default=1)
parser.add_argument('--train-split', '-ts', type=float, default=0.8, help='Portion of data used as training data')
parser.add_argument('--num-epochs', '-ne', type=int, default=100, help='Number of epochs used to train the model')

# Network configuration
parser.add_argument('--partition-shape', '-ps', type=int, nargs='+', help='Partition shape used throughout the network')
parser.add_argument('--width', '-w', type=int, default=20, help='Channel dimension of lifted space')
parser.add_argument('--modes', '-m', type=int, nargs='+', default=None, help='Number of modes to take in each dimension. Defaults to d//4')
parser.add_argument('--num-blocks', '-nb', type=int, default=4, help='Number of spectral convolution blocks in the FNO')

args = parser.parse_args()

# Create partitions
dim = len(args.partition_shape)
nw = np.prod(args.partition_shape)

P_world = Partition(MPI.COMM_WORLD)
P_0_base = P_world.create_partition_inclusive([0])
P_0 = P_0_base.create_cartesian_topology_partition([1]*dim)
P_x_base = P_world.create_partition_inclusive(np.arange(nw))
P_x = P_x_base.create_cartesian_topology_partition(args.partition_shape)

# Create simple basic communication primitives
B = distdl.nn.Broadcast(P_0, P_x)
S = distdl.nn.DistributedTranspose(P_0, P_x)
G = distdl.nn.DistributedTranspose(P_x, P_0)

# -- Preprocess --
timer = Timer(P_x)
timer.start('preprocess')

# Create data output
timestamp = zero_volume_tensor()
if P_0.active:
    timestamp = Tensor([int(time.time())])

timestamp = B(timestamp)
timestamp = int(timestamp)

output_dir = Path(f'data/{args.input.stem}-{timestamp}')
info_path = output_dir.joinpath(Path('info.txt'))
if P_0.active:
    os.makedirs(output_dir)
    print(f'Created data output directory: {output_dir}')

    with open(info_path, 'w') as f:
        f.write(f'{args}\n')

    print(f'Wrote info file: {info_path}')

P_world._comm.Barrier()

# Load mat data. TODO: Do this using MPIIO
if P_0.active:

    print(f'Loading mat data: {args.input}...')
    mat = loadmat(args.input)
    print('Done.')

    a = mat['a']
    u = mat['u']
    t = mat['t']
    
    print('== mat ==')
    print(f'a.shape = {a.shape}')
    print(f'u.shape = {u.shape}')
    print(f't.shape = {t.shape}')

    print(f'Splitting data into train/test...')
    
    # Do dim-3 here because the .mat data provided has no channel dimension
    split_idx = int(u.shape[0] * args.train_split)
    X_train_slice = tuple([slice(0, 80, 1), *[slice(0, None, args.subsampling_rate)]*(dim-3), slice(0, args.input_steps, 1)])
    Y_train_slice = tuple([slice(0, 80, 1), *[slice(0, None, args.subsampling_rate)]*(dim-3), slice(args.input_steps, None, 1)])
    X_test_slice  = tuple([slice(80, 100, 1), *[slice(0, None, args.subsampling_rate)]*(dim-3), slice(0, args.input_steps, 1)])
    Y_test_slice  = tuple([slice(80, 100, 1), *[slice(0, None, args.subsampling_rate)]*(dim-3), slice(args.input_steps, None, 1)])

    X_train = Tensor(u[X_train_slice]).unsqueeze(1)
    Y_train = Tensor(u[Y_train_slice]).unsqueeze(1)
    X_test  = Tensor(u[X_test_slice]).unsqueeze(1)
    Y_test  = Tensor(u[Y_test_slice]).unsqueeze(1)
    
    grid_train = create_grid(X_train.shape)
    grid_test  = create_grid(X_test.shape)

    print(f'X_train.shape    = {X_train.shape}')
    print(f'Y_train.shape    = {Y_train.shape}')
    print(f'X_test.shape     = {X_test.shape}')
    print(f'Y_test.shape     = {Y_test.shape}')
    print(f'grid_train.shape = {grid_train.shape}')
    print(f'grid_test.shape  = {grid_test.shape}')

else:
    X_train = zero_volume_tensor()
    Y_train = zero_volume_tensor()
    X_test  = zero_volume_tensor()
    Y_test  = zero_volume_tensor()

    grid_train = zero_volume_tensor() 
    grid_test  = zero_volume_tensor() 

P_world._comm.Barrier()

# Scatter train/test data and grid
X_train = S(X_train)
P_world._comm.Barrier()
S = distdl.nn.DistributedTranspose(P_0, P_x)
Y_train = S(Y_train)
P_world._comm.Barrier()
S = distdl.nn.DistributedTranspose(P_0, P_x)
X_test = S(X_test)
P_world._comm.Barrier()
S = distdl.nn.DistributedTranspose(P_0, P_x)
Y_test = S(Y_test)
P_world._comm.Barrier()
S = distdl.nn.DistributedTranspose(P_0, P_x)

grid_train = S(grid_train)
P_world._comm.Barrier()
S = distdl.nn.DistributedTranspose(P_0, P_x)
grid_test  = S(grid_test)
P_world._comm.Barrier()
S = distdl.nn.DistributedTranspose(P_0, P_x)

X_train.requires_grad = True

print(f'rank = {P_x.rank:04d}, X_train.shape    = {X_train.shape}')
print(f'rank = {P_x.rank:04d}, Y_train.shape    = {Y_train.shape}')
print(f'rank = {P_x.rank:04d}, X_test.shape     = {X_test.shape}')
print(f'rank = {P_x.rank:04d}, Y_test.shape     = {Y_test.shape}')
print(f'rank = {P_x.rank:04d}, grid_train.shape = {grid_train.shape}')
print(f'rank = {P_x.rank:04d}, grid_test.shape  = {grid_test.shape}')

X_train = torch.cat((X_train, grid_train), dim=1)
X_test = torch.cat((X_test, grid_test), dim=1)

train_set = TensorDataset(X_train, Y_train)
test_set  = TensorDataset(X_test, Y_test)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader  = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

timer.stop('preprocess', f'{P_x.rank}')

if P_0.active:
    print('Done.')

# Setup network, loss function, and optimizer
in_channels = X_train.shape[1]
out_timesteps = Y_test.shape[-1]
network = DistributedFNONd(P_x, in_channels, out_timesteps, args.modes, args.width, args.num_blocks)
parameters = [p for p in network.parameters()]
criterion = distdl.nn.DistributedMSELoss(P_x)
optimizer = torch.optim.Adam(parameters, lr=1e-3)

# Train network
if P_0.active:
    print('Training network...')

timings_out_path = output_dir.joinpath(Path(f'timings-{P_x.rank}.txt'))

timer.start('train')

for m in range(args.num_epochs):
    
    network.train()

    for n, (X, Y) in enumerate(train_loader):
        
        timer.start('batch')

        optimizer.zero_grad()

        timer.start('encode')
        X_enc, mu_x, sigma_x = unit_gaussian_encode(P_x, P_0, X)
        Y_enc, mu_y, sigma_y = unit_gaussian_encode(P_x, P_0, Y)
        timer.stop('encode', f'{m}, {n}, {P_x.rank}')

        timer.start('forward')
        Y_hat = network(X_enc)
        timer.stop('forward', f'{m}, {n}, {P_x.rank}')

        timer.start('decode')
        Y_dec = unit_gaussian_decode(Y_hat, mu_y, sigma_y)
        timer.stop('decode', f'{m}, {n}, {P_x.rank}')

        timer.start('loss')
        loss = criterion(Y, Y_dec)
        timer.stop('loss', f'{m}, {n}, {P_x.rank}')

        if P_0.active:
            print(f'epoch = {m}, batch = {n}, loss = {loss.item()}')

        timer.start('adjoint')
        loss.backward()
        timer.stop('adjoint', f'{m}, {n}, {P_x.rank}')

        timer.start('step')
        optimizer.step()
        timer.stop('step', f'{m}, {n}, {P_x.rank}')

        timer.stop('batch', f'{m}, {n}, {P_x.rank}') 
        
        with open(timings_out_path, 'a') as f:
            timer.dump_times(f)

    network.eval()
    error = 0.0
    with torch.no_grad():
        for n, (X, Y) in enumerate(test_loader):
            X_enc, mu_x, sigma_x = unit_gaussian_encode(P_x, P_0, X)
            Y_enc, mu_y, sigma_y = unit_gaussian_encode(P_x, P_0, Y)
            Y_hat = network(X_enc)
            Y_dec = unit_gaussian_decode(Y_hat, mu_y, sigma_y)
            loss = criterion(Y, Y_dec)
            error += loss.item()

        if P_0.active:
            print(f'epoch = {m}, test error = {error}')

    # timer.stop('epoch', f'{m}, {P_x.rank}')

    with open(timings_out_path, 'a') as f:
        timer.dump_times(f)

# timer.stop('train', f'{P_x.rank}')
print('Done.')

# Switch model to eval mode
network.eval()

# Save the model
network_path = output_dir.joinpath(Path(f'network-{P_x.rank}.pt'))
# torch.save(network, network_path)
# print(f'rank = {P_x.rank}, wrote model to path: {network_path}')

# Evaluate test set
Y_pred = []

with torch.no_grad():
    for n, (X, Y) in enumerate(test_loader):
        X_enc, mu_x, sigma_x = unit_gaussian_encode(P_x, P_0, X)
        Y_enc, mu_y, sigma_y = unit_gaussian_encode(P_x, P_0, Y)
        Y_hat = network(X_enc)
        Y_dec = unit_gaussian_decode(Y_hat, mu_y, sigma_y)
        Y_pred.append(Y_dec)

Y_pred = torch.cat(tuple(Y_pred))

mat_path = output_dir.joinpath(Path(f'test-pred-mat-{P_x.rank}.mat'))
print(f'Writing predictions to file: {mat_path}...')
io.savemat(mat_path, mdict={'x_test': X_test.cpu().numpy(), 'y_test': Y_test.cpu().numpy(), 'y_pred': Y_pred.cpu().numpy()})
print('Done.')
