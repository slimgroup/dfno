from dfno import create_standard_partitions, DistributedFNO
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from pathlib import Path

import json
import gc
import multiprocessing
import os, sys
import time
import torch
import traceback
from mpi4py import MPI
import cupy

from dfno import profile_gpu_memory, compute_distribution_info

def dls(l, delimiter='_'):
    out = ''
    for i, x in enumerate(l):
        out += str(x)
        if i < len(l)-1:
            out += delimiter
    return out

def print0(x, P_0):
    if P_0.active:
        print(x)
        sys.stdout.flush()

def bench(input_shape, partition_shape, width, modes, nt, dev, ngpu, benchmark_type, fft_order=None, output_dir=Path('.')):

    P_world, P_x, P_0 = create_standard_partitions(partition_shape)
    if dev == 'cpu':
        device_name = 'cpu'
    else:
        device_ordinal = P_x.rank % ngpu
        device_name = f'cuda:{device_ordinal}'

    device = torch.device(device_name)
    if fft_order is None:
        outfile = Path(f'{dls(input_shape)}-{dls(partition_shape)}-{width}-{dls(modes)}-{nt}-{benchmark_type}-default-ordering-{P_x.rank}-{P_x.size}.json')
    else:
        fft_order_before, fft_order_after = fft_order
        outfile = Path(f'{dls(input_shape)}-{dls(partition_shape)}-{width}-{dls(modes)}-{nt}-{benchmark_type}-{dls(fft_order_before)}-{dls(fft_order_after)}-{P_x.rank}-{P_x.size}.json')
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
    bench_gpu_mem = False
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

        def bench_inner():
            print0("initialize", P_0)
            x = torch.rand(size=tuple(x_info['shape']), device=device, dtype=torch.float32)
            network = DistributedFNO(P_x, x_shape, nt, width, modes, fft_order=fft_order, device=device, dtype=torch.float32)
            network.eval()


            if benchmark_type == 'eval':
                with torch.no_grad():
                    print0("fake eval", P_0)
                    y = network(x)
                    print0("fake eval done", P_0)
                    del y
                    gc.collect()

                    P_x._comm.Barrier()
                    print0("real eval", P_0)
                    t0 = time.time()
                    y = network(x)
                    t1 = time.time()
                    print0("real eval done", P_0)
                    data['dt'] = t1-t0
                    data['dt_comm'] = network.dt_comm
                    data['dt_comp'] = data['dt'] - data['dt_comm']

            else:
                print0("fake eval", P_0)
                y = network(x)
                y1 = torch.ones_like(y)
                print0("fake grad", P_0)
                y.backward(y1)
                print0("fake grad done", P_0)
                del y
                gc.collect()

                P_x._comm.Barrier()
                print0("real eval", P_0)
                t0 = time.time()
                y = network(x)
                t1 = time.time()
                print0("real eval done", P_0)
                data['dt'] = t1-t0
                data['dt_comm'] = network.dt_comm
                data['dt_comp'] = data['dt'] - data['dt_comm']

                P_x._comm.Barrier()
                print0("real grad", P_0)
                t0 = time.time()
                y.backward(y1)
                t1 = time.time()
                print0("real grad done", P_0)
                data['dt_grad'] = t1-t0

        if dev == 'cpu':
            bench_inner()
        else:
            with cupy.cuda.Device(device_ordinal):
                bench_inner()

        with open(output_dir.joinpath(outfile), 'w') as f:
            json.dump(data, f)

    except:
        # catch errors to avoid hanging
        traceback.print_exc()
        errors = True

    if bench_gpu_mem:
        sys.stdout.flush()
        proc.terminate()
        proc.kill()
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
    parser.add_argument('--fft-order-before', type=int, nargs='+')
    parser.add_argument('--fft-order-after', type=int, nargs='+')
    parser.add_argument('--mydummyargument', nargs='?', required=False)

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
    fft_order = None
    if args.fft_order_before is not None and args.fft_order_after is not None:
        fft_order = (args.fft_order_before, args.fft_order_after)
    elif args.fft_order_before is not None:
        # only "before" is specified; "after" is whatever's left
        after = list(range(2,len(args.input_shape)))
        after = [ x for x in after if x not in args.fft_order_before ]
        fft_order = (args.fft_order_before, after)
    elif args.fft_order_after is not None:
        # only "after" is specified; "before" is whatever's left
        before = list(range(2,len(args.input_shape)))
        before = [ x for x in before if x not in args.fft_order_after ]
        fft_order = (before, args.fft_order_after)

    bench(input_shape, partition_shape, width, modes, nt, device, ngpu, benchmark_type, fft_order=fft_order, output_dir=output_dir)
