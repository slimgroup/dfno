from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import os

def format_runs(runs, data_dir, shape, modes, run_type, mode='spatial'):
    out = r'#!/bin/bash' + '\nset -x' + '\n\n'
    out += f'data_dir={data_dir}\n'

    for nprocs, partition_shape in runs:
        out += f'jsrun -n {nprocs} --cpu_per_rs=7 --rs_per_host={min(nprocs, 6)} --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is '
        #out += f'mpirun -np {nprocs} python3 bench.py -is '
        
        if mode == 'spatial':
            shape_np = [s*ps for s, ps in zip(shape, partition_shape[:-1])]
            shape_np.append(1)
            modes_np = [m*ps for m, ps in zip(modes, partition_shape[2:])]
            nt = shape[-1]*partition_shape[-1]
        else:
            shape_np = [*shape[:-1], 1]
            modes_np = [*modes[:-1], np.prod(partition_shape)*modes[-1]]
            nt = np.prod(partition_shape)*shape[-1]

        for x in shape_np:
            out += f'{x} '

        out += '-m '
        for x in modes_np:
            out += f'{x} '
        
        out += '-ps '
        for x in partition_shape:
            out += f'{x} '

        out += f'-w 20 -nt {nt} -d cuda -ngpu 4 -bt {run_type} -o $data_dir\n'

    return out

def create_script(name, runs, shape, modes, run_type, mode='spatial'):
    data_dir = Path(name)
    fname = Path(f'{name}.sh')
    with open(fname, 'w') as f:
        out = format_runs(runs, data_dir, shape, modes, run_type, mode=mode)
        f.write(out)
    os.chmod(fname, 0o755)
    print(f'created script: {fname.resolve()}')

local_shape = (1, 1, 64, 64, 64, 16)
local_modes = (4, 4, 4, 4)
runs = [
    (1,    (1, 1, 1,  1,  1,  1)),
    (2,    (1, 1, 2,  1,  1,  1)),
    (4,    (1, 1, 2,  2,  1,  1)),
    (6,    (1, 1, 3,  2,  1,  1)),
    (12,   (1, 1, 3,  2,  2,  1)),
    (24,   (1, 1, 4,  3,  2,  1)),
    (48,   (1, 1, 4,  4,  3,  1)),
    (96,   (1, 1, 6,  4,  4,  1)),
    (192,  (1, 1, 8,  6,  4,  1)),
    (384,  (1, 1, 8,  8,  6,  1)),
    (768,  (1, 1, 12, 8,  8,  1)),
]

scripts_spatial = [
    ('eval_weak_scaling_spatial_gpu', 'eval'),
    ('grad_weak_scaling_spatial_gpu', 'grad'),
]

scripts_temporal = [
    ('eval_weak_scaling_temporal_gpu', 'eval'),
    ('grad_weak_scaling_temporal_gpu', 'grad'),
]

parser = ArgumentParser()
parser.add_argument('--max-workers', '-mw', type=int, default=-1)
parser.add_argument('--clean-old', '-co', action='store_true')
args = parser.parse_args()

if args.max_workers > 0:
    runs_spatial = [x for x in runs if x[0] <= args.max_workers]
    runs_temporal = [x for x in runs if x[0] <= args.max_workers]
else:
    runs_spatial = runs
    runs_temporal = runs

if args.clean_old:
    print('removing existing scripts...')
    for fname in os.listdir(Path('.')):
        fpath = Path(fname)
        if fpath.suffix == '.sh':
            os.remove(fpath.resolve())
            print(f'removed script: {fpath.resolve()}')

for ss in scripts_spatial:
    create_script(ss[0], runs_spatial, local_shape, local_modes, ss[1], mode='spatial')

for st in scripts_temporal:
    create_script(st[0], runs_temporal, local_shape, local_modes, st[1], mode='temporal')
