from argparse import ArgumentParser
from dfno import create_standard_partitions, DistributedFNONd
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from pathlib import Path

import distdl.nn as dnn
import numpy as np
import os
import time
import torch

def compute_distribution_info(P, shape):
    info = {}
    ts = TensorStructure()
    ts.shape = shape
    info['shapes'] = compute_subtensor_shapes_balanced(ts, P.shape)
    info['starts'] = compute_subtensor_start_indices(info['shapes'])
    info['stops']  = compute_subtensor_stop_indices(info['shapes'])
    info['index']  = tuple(P.index)
    info['shape']  = info['shapes'][info['index']]
    info['start']  = info['starts'][info['index']]
    info['stop']   = info['stops'][info['index']]
    info['slice']  = assemble_slices(info['start'], info['stop'])
    return info

parser = ArgumentParser()
parser.add_argument('--input-shape', '-is', type=int, nargs='+')
parser.add_argument('--output-shape', '-os', type=int, nargs='+')
parser.add_argument('--partition-shape', '-ps', type=int, nargs='+')
parser.add_argument('--num-gpus', '-ngpu', type=int, default=0)
parser.add_argument('--data-dir', type=Path, default=Path('data'))
parser.add_argument('--device', '-d', type=str, default='cpu')
parser.add_argument('--width', '-w', type=int, default=20)
parser.add_argument('--modes', '-m', type=int, nargs='+')
parser.add_argument('--num_iter', '-n', type=int, default=10)

args = parser.parse_args()
input_shape = np.array(args.input_shape, dtype=int)
output_shape = np.array(args.output_shape, dtype=int)
partition_shape = np.array(args.partition_shape, dtype=int)
num_gpus = args.num_gpus
data_dir = args.data_dir
device = torch.device(args.device)
width = args.width
modes = np.array(args.modes)
n = args.num_iter

assert len(input_shape) == len(output_shape) == len(partition_shape)
assert input_shape[0] == output_shape[0]
assert output_shape[1] == 1
assert np.array_equal(input_shape[2:-1], output_shape[2:-1])
assert input_shape[-1] % 2 == 0 # TODO: fix
assert width > 0
assert len(input_shape) == len(modes) + 2
assert n > 0

P_world, P_x, P_0 = create_standard_partitions(partition_shape)

B = dnn.Broadcast(P_0, P_x)
timestamp = torch.tensor([int(time.time())]) if P_0.active else zero_volume_tensor()
timestamp = B(timestamp).item()
out_dir = data_dir.joinpath(str(timestamp))

if P_0.active:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Created output directory: {out_dir.resolve()}')

P_x._comm.Barrier()

info_x = compute_distribution_info(P_x, input_shape)
info_y = compute_distribution_info(P_x, output_shape)

x = torch.rand(tuple(info_x['shape']), device='cpu', dtype=torch.float32)
y = torch.rand(tuple(info_y['shape']), device='cpu', dtype=torch.float32)

network = DistributedFNONd(P_x, width, modes, output_shape[-1], device='cpu')
with torch.no_grad():
    _ = network(x)

network = network.to(device)
x = x.to(device)
y = y.to(device)
criterion = dnn.DistributedMSELoss(P_x).to(device)

with torch.profiler.profile(profile_memory=True, with_stack=True) as p:
    for i in range(n):
        print(i)
        y_hat = network(x)
        loss = criterion(y, y_hat)
        loss.backward()
        p.step()

stacks_path = out_dir.joinpath(f'profiler_{P_x.rank:04d}.stacks')
p.export_stacks(stacks_path, metric='self_cpu_time_total')
print(p.key_averages().table())
