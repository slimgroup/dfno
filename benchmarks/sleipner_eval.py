from dfno import create_standard_partitions, DistributedFNONd
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from pathlib import Path

import distdl.nn as dnn
import gc
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

ps = (1, 1, 1, 2, 2, 1)
P_world, P_x, P_0 = create_standard_partitions(ps)

ngpu = 4
device = torch.device(f'cuda:{P_x.rank % ngpu}')
data_dir = Path('.')

b = 1
c = 1
nx = 64
ny = 64
nz = 64
nt = 10

w = 20
m = (4, 4, 4, 4)

x_shape = (b, c, nx, ny, nz, 1)
y_shape = (b, c, nx, ny, nz, nt)
x_info = compute_distribution_info(P_x, x_shape)
y_info = compute_distribution_info(P_x, y_shape)

x = torch.rand(size=tuple(x_info['shape']), device=torch.device('cpu'), dtype=torch.float32)
network = DistributedFNONd(P_x, w, m, nt, device='cpu')
network.eval()

with torch.no_grad():
    _ = network(x)
gc.collect()

network.to(device)
