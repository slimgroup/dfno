from dfno import create_standard_partitions, DistributedFNONd
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from pathlib import Path

import distdl.nn as dnn
import json
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

def dls(l, delimiter='_'):
    out = ''
    for i, x in enumerate(l):
        out += str(x)
        if i < len(l)-1:
            out += delimiter
    return out


def bench(input_shape, partition_shape, width, modes, nt, ngpu, benchmark_type, output_dir=Path('.')):

    P_world, P_x, P_0 = create_standard_partitions(partition_shape)
    device = torch.device('cpu') if args.device == 'cpu' else torch.device(f'cuda:{P_x.rank % ngpu}')
    outfile = Path(f'{dls(input_shape)}-{dls(partition_shape)}-{width}-{dls(modes)}-{nt}-{benchmark_type}-{P_x.rank}-{P_x.size}.json')
    data = {}

    assert len(input_shape) == len(partition_shape)
    assert width > 0
    assert len(input_shape)-2 == len(modes)
    assert nt > 0

    if not os.path.exists(output_dir) and P_0.active:
        os.makedirs(output_dir)
        print(f'created output directory: {output_dir}')

    P_x._comm.Barrier()

    x_shape = input_shape
    y_shape = (*input_shape[:-1], nt)
    x_info = compute_distribution_info(P_x, x_shape)

    network = DistributedFNONd(P_x, width, modes, nt, device='cpu')
    network.eval()

    dummy = torch.rand(size=tuple(x_info['shape']), device=torch.device('cpu'), dtype=torch.float32)
    y = network(dummy)
    del dummy
    del y
    gc.collect()

    network.to(device)
    P_x._comm.Barrier()
    x = torch.rand(size=tuple(x_info['shape']), device=device, dtype=torch.float32)

    if benchmark_type == 'eval':
        with torch.no_grad():
            P_x._comm.Barrier()
            t0 = time.time()
            y = network(x)
            t1 = time.time()
            data['dt'] = t1-t0

    else:
        P_x._comm.Barrier()
        t0 = time.time()
        y = network(x)
        t1 = time.time()
        data['dt'] = t1-t0

        y1 = torch.ones_like(y)
        P_x._comm.Barrier()
        t0 = time.time()
        y.backward(y1)
        t1 = time.time()
        data['dt_grad'] = t1-t0

    with open(output_dir.joinpath(outfile), 'w') as f:
        json.dump(data, f)

if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--input-shape', '-is', type=int, nargs='+')
    parser.add_argument('--partition_shape', '-ps', type=int, nargs='+')
    parser.add_argument('--width', '-w', type=int, default=20)
    parser.add_argument('--modes', '-m', type=int, nargs='+')
    parser.add_argument('--num-timesteps', '-nt', type=int, default=10)
    parser.add_argument('--device', '-d', type=str, default='cpu')
    parser.add_argument('--num-gpus', '-ngpu', type=int, default=0)
    parser.add_argument('--benchmark-type', '-bt', type=str, default='eval')
    parser.add_argument('--output-dir', '-o', type=Path, default=Path('.'))

    args = parser.parse_args()
    input_shape = args.input_shape
    partition_shape = args.partition_shape
    width = args.width
    modes = args.modes
    nt = args.num_timesteps
    ngpu = args.num_gpus
    benchmark_type = args.benchmark_type
    output_dir = args.output_dir

    bench(input_shape, partition_shape, width, modes, nt, ngpu, benchmark_type, output_dir)
