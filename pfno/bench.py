from pfno import create_standard_partitions, ParallelFNO
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from pathlib import Path

import distdl.nn as dnn
import json
import gc
import multiprocessing
import numpy as np
import os, sys
import time
import torch
import traceback
from mpi4py import MPI
import cupy

from utils import profile_gpu_memory

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


def bench(input_shape, partition_shape, width, modes, nt, dev, ngpu, benchmark_type, output_dir=Path('.')):

    P_world, P_x, P_0 = create_standard_partitions(partition_shape)
    device_ordinal = P_x.rank % ngpu

    device_name = 'cpu' if dev == 'cpu' else f'cuda:{device_ordinal}'
    device = torch.device(device_name)
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

    bench_gpu_mem = True if 'cuda' in device_name else False
    if bench_gpu_mem:
        outfile_mem = output_dir.joinpath(Path(f'{dls(input_shape)}-{dls(partition_shape)}-{width}-{dls(modes)}-{nt}-{benchmark_type}-{P_x.rank}-{P_x.size}_mem.json'))
        proc = multiprocessing.Process(target=profile_gpu_memory, args=(outfile_mem, 0.25))
        proc.daemon = True
        time.sleep(5)
        proc.start()

    P_x._comm.Barrier()
    errors = False
    
    try:
        x_shape = input_shape
        y_shape = (*input_shape[:-1], nt)
        x_info = compute_distribution_info(P_x, x_shape)

        x = torch.rand(size=tuple(x_info['shape']), device=device, dtype=torch.float32)
        network = ParallelFNO(P_x, x_shape, nt, width, modes, device=device, dtype=torch.float32)
        network.eval()
        
        with cupy.cuda.Device(device_ordinal):
            if benchmark_type == 'eval':
                with torch.no_grad():
                    P_x._comm.Barrier()
                    t0 = time.time()
                    y = network(x)
                    t1 = time.time()
                    data['dt'] = t1-t0
                    data['dt_comm'] = network.dt_comm
                    data['dt_comp'] = data['dt'] - data['dt_comm']

            else:
                P_x._comm.Barrier()
                t0 = time.time()
                y = network(x)
                t1 = time.time()
                data['dt'] = t1-t0
                data['dt_comm'] = network.dt_comm
                data['dt_comp'] = data['dt'] - data['dt_comm']

                y1 = torch.ones_like(y)
                P_x._comm.Barrier()
                t0 = time.time()
                y.backward(y1)
                t1 = time.time()
                data['dt_grad'] = t1-t0

        with open(output_dir.joinpath(outfile), 'w') as f:
            json.dump(data, f)

    except:
        # catch errors to avoid hanging
        traceback.print_exc()
        errors = True

    if bench_gpu_mem:
        print("Terminating memory profiler")
        sys.stdout.flush()
        proc.terminate()
        proc.kill()
        print("Terminated memory profiler")
        if errors: MPI.COMM_WORLD.Abort(1)

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
    device = args.device
    ngpu = args.num_gpus
    benchmark_type = args.benchmark_type
    output_dir = args.output_dir

    bench(input_shape, partition_shape, width, modes, nt, device, ngpu, benchmark_type, output_dir)
