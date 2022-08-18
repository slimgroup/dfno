from argparse import ArgumentParser
from pathlib import Path

from math import ceil
import numpy as np
import os

parser = ArgumentParser()
parser.add_argument('--system', default="local", choices=["local", "summit", "perlmutter"])
parser.add_argument('--max-workers', '-mw', type=int, default=-1)
parser.add_argument('--clean-old', '-co', action='store_true')
args = parser.parse_args()

def format_runs(name, fname, runs, data_dir, shape, modes, run_type, mode='spatial'):
    all_runs = []
    out = r'#!/bin/bash' + '\nset -x' + '\n\n'
    out += f'data_dir={data_dir}\n'
    out += ("if test \"x$1\" = x; then\n"
            "  echo \"Usage: $0 <numranks>\"\n"
            "  exit 0\n"
            "fi\n"
            "ranks=$1\n")

    for nprocs, partition_shape in runs:
        out += f"[[ $ranks -eq '{nprocs}' ]] && "
        if args.system == 'summit':
            PWD=os.getcwd()
            # summit has 6 GPUs per compute node
            out += f'jsrun -n {nprocs} --cpu_per_rs=7 --rs_per_host={min(nprocs, 6)} --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs --smpiargs="-gpu" python3 bench.py --input-shape '
            #out += f'jsrun -n {nprocs} --cpu_per_rs=7 --rs_per_host={min(nprocs, 6)} --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs --smpiargs="-gpu" ./bench.sh $data_dir --input-shape '
            nnodes = ceil(nprocs / 6.0)
            all_runs.append(f'bsub -W 0:30 -nnodes {nnodes} -P $MYPROJ -q batch -J DFNObench.{nprocs} {PWD}/launch-summit.sh {nprocs}')
        elif args.system == 'perlmutter':
            # perlmutter has 4 GPUs per compute node
            out += f'PYTHONPATH=$PWD/.. srun --ntasks={nprocs} --ntasks-per-node={min(nprocs, 4)} --cpus-per-task=32 --gpus-per-task=1 --gpu-bind=per_task:1 python3 bench.py --input-shape '
            nnodes = ceil(nprocs / 4.0)
            all_runs.append(f'sbatch --account=$ACCOUNT --constraint=gpu --nodes={nnodes} --qos=regular --time=0:30:00 --job-name={name}.{nprocs} {fname} {nprocs}')
        else:
            # run locally (development)
            out += f'mpirun -np {nprocs} python3 bench.py --input-shape '

        if mode == 'spatial':
            shape_np = [s*ps for s, ps in zip(shape, partition_shape[:-1])]
            shape_np.append(1)
            modes_np = [m*ps for m, ps in zip(modes, partition_shape[2:])]
            nt = shape[-1]*partition_shape[-1]
        else:
            shape_np = [*shape[:-1], 1]
            modes_np = [*modes[:-1], np.prod(partition_shape)*modes[-1]]
            nt = np.prod(partition_shape)*shape[-1]

        shape_in = [*shape_np[:-1], nt]
        p1, p2 = partition_shape[2]*partition_shape[4], partition_shape[3]*partition_shape[5]
        if p1 > shape_in[2]:
            raise Exception(f'Invalid configuration. Partitions {partition_shape} and input shape {shape_in} would produce zero-size at dimension 2')
        if p1 > shape_in[4]:
            raise Exception(f'Invalid configuration. Partitions {partition_shape} and input shape {shape_in} would produce zero-size at dimension 4')
        if p2 > shape_in[3]:
            raise Exception(f'Invalid configuration. Partitions {partition_shape} and input shape {shape_in} would produce zero-size at dimension 3')
        if p2 > shape_in[5]//2:
            raise Exception(f'Invalid configuration. Partitions {partition_shape} and input shape {shape_in} would produce zero-size at dimension 5')

        for x in shape_np:
            out += f'{x} '

        out += '--modes '
        for x in modes_np:
            out += f'{x} '

        out += '--partition_shape '
        for x in partition_shape:
            out += f'{x} '

        out += f'--width 20 --num-timesteps {nt} --device cuda --num-gpus 1 --benchmark-type {run_type} --output-dir $data_dir\n'

    return out, all_runs

def create_runscript(name, runs, shape, modes, run_type, mode='spatial'):
    print(runs)
    data_dir = Path(name)
    fname = Path(f'{name}.sh')
    with open(fname, 'w') as f:
        out, all_runs = format_runs(name, fname, runs, data_dir, shape, modes, run_type, mode=mode)
        f.write(out)
    os.chmod(fname, 0o755)
    print(f'created script for {args.system}: {fname.name}')
    return all_runs

def create_launchscript(fname, things_to_run):
    out = "#!/bin/bash\n"
    if args.system == 'summit':
        out += """
if test x$MYPROJ = x; then
    echo Please set MYPROJ to your Summit project ID.
    echo Example: export MYPROJ=csc471
    exit 1
fi
"""
    elif args.system == 'perlmutter':
        out += f"""
if test x$ACCOUNT = x; then
    echo Please set ACCOUNT to the GPU account for your project.
    echo Example: export ACCOUNT=m3863_g
    exit 1
fi
"""
    out += "\nset -x\n\n"

    out += "\n".join(things_to_run)
    out += "\n"
    fname = Path(fname)
    with open(fname, 'w') as f:
        f.write(out)
    os.chmod(fname, 0o755)
    print(f'created batch submission script: {fname.name}')

local_modes = (4, 4, 4, 4)
if args.system == 'summit':
    local_shape = (1, 1, 48, 48, 48, 32)
    # summit has 6 GPUs per compute node
    runs = [
        (1,      (1, 1, 1,  1,  1,  1)), # ⅙ of 1 node
        (2,      (1, 1, 2,  1,  1,  1)), # ⅓ of 1 node
        (4,      (1, 1, 2,  2,  1,  1)), # ⅔ of 1 node
        (6,      (1, 1, 3,  2,  1,  1)), # 1 node
        (12,     (1, 1, 3,  2,  2,  1)), # 2 nodes
        (24,     (1, 1, 4,  3,  2,  1)), # 4 nodes
        (48,     (1, 1, 4,  4,  3,  1)), # 8 nodes
        (96,     (1, 1, 6,  4,  4,  1)), # 16 nodes
        (192,    (1, 1, 8,  6,  4,  1)), # 32 nodes
        (384,    (1, 1, 8,  8,  6,  1)), # 64 nodes
        (768,    (1, 1, 12, 8,  8,  1)), # 128 nodes
        (1536,   (1, 1, 16, 8, 12,  1)), # 256 nodes
        (3072,   (1, 1, 32, 8, 12,  1)), # 512 nodes
        (6144,   (1, 1, 24, 8, 32,  1)), # 1024 nodes
        (12288,  (1, 1, 32, 8, 48,  1)), # 2048 nodes
    ]
elif args.system == 'perlmutter':
    local_shape = (1, 1, 64, 64, 64, 32)
    # perlmutter has 4 GPUs per compute node
    runs = [
        (1,      (1, 1, 1,  1,  1,  1)), # ¼ of 1 node
        (2,      (1, 1, 2,  1,  1,  1)), # ½ of 1 node
        (4,      (1, 1, 2,  2,  1,  1)), # 1 node
        (8,      (1, 1, 2,  2,  2,  1)), # 2 nodes
        (16,     (1, 1, 4,  2,  2,  1)), # 4 nodes
        (32,     (1, 1, 4,  4,  2,  1)), # 8 nodes
        (64,     (1, 1, 4,  4,  4,  1)), # 16 nodes
        (128,    (1, 1, 8,  4,  4,  1)), # 32 nodes
        (256,    (1, 1, 8,  8,  4,  1)), # 64 nodes
    ]
elif args.system == "local":
    # with this size, 4 rank spatial with gradient takes about 5.5GB of GPU memory
    local_shape = (1, 1, 32, 16, 16, 10)
    runs = [
        (1,      (1, 1, 1,  1,  1,  1)), # ~1.4GB of GPU memory (with gradient)
        (2,      (1, 1, 2,  1,  1,  1)), # ~2.7GB of GPU memory (with gradient)
        (4,      (1, 1, 2,  1,  2,  1)), # ~5.5GB of GPU memory (with gradient)
    ]
else:
    raise Exception(f"invalid value '{args.system}' for parameter 'system'")

scripts_spatial = [
    ('eval_weak_scaling_spatial_gpu', 'eval'),
    ('grad_weak_scaling_spatial_gpu', 'grad'),
]

scripts_temporal = [
    ('eval_weak_scaling_temporal_gpu', 'eval'),
    ('grad_weak_scaling_temporal_gpu', 'grad'),
]

if args.max_workers > 0:
    runs_spatial = [x for x in runs if x[0] <= args.max_workers]
    runs_temporal = [x for x in runs if x[0] <= args.max_workers and x[0] < 768]
else:
    runs_spatial = runs
    runs_temporal = [x for x in runs if x[0] < 768]

if args.clean_old:
    print('removing existing scripts...')
    generated_scripts = set()
    for scaling_type in [scripts_spatial, scripts_temporal]:
        for basename, exec_type in scaling_type:
            generated_scripts.add(basename + ".sh")
    for fname in os.listdir(Path('.')):
        fpath = Path(fname)
        if fpath.name in generated_scripts:
            os.remove(fpath.resolve())
            print(f'removed script: {fpath.resolve()}')

things_to_run = []
for name, runtype in scripts_spatial:
    things_to_run += create_runscript(name, runs_spatial, local_shape, local_modes, runtype, mode='spatial')

for name, runtype in scripts_temporal:
    things_to_run += create_runscript(name, runs_temporal, local_shape, local_modes, runtype, mode='temporal')

# summit runs launch-summit.sh for multiple configurations, and launch-summit.sh runs them all.  consolidate the submit commands.
unique_runs = []
runset = set()
for run in things_to_run:
    if run in runset:
        continue
    runset.add(run)
    unique_runs.append(run)

if args.system != "local":
    create_launchscript(f"submit_{args.system}.sh", unique_runs)
