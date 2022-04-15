import subprocess as sp
import os
import time
import torch
import distdl

from contextlib import nullcontext
from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from typing import Any, List, Tuple, Dict

Partition = distdl.backend.backend.Partition

# https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        tmp = sp.check_output(COMMAND,stderr=sp.STDOUT,shell=True)
        memory_use_info = output_to_list(tmp)[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

def profile_gpu_memory(outfile, dt=1.0):
    t0 = time.time()
    with open(outfile, 'w') as f:
        while True:
            muv = get_gpu_memory()
            f.write(f'{str(time.time()-t0)}, ')
            for i, m in enumerate(muv):
                f.write(str(m))
                if i < len(muv)-1:
                    f.write(', ')
            f.write('\n')
            f.flush()
            time.sleep(dt)

def get_env(P, num_gpus=1):

    cuda_aware = 'CUDA_AWARE' in os.environ
    use_cuda = 'USE_CUDA' in os.environ or cuda_aware
    device_ordinal = P.rank % num_gpus
    device = torch.device(f'cuda:{device_ordinal}') if use_cuda else torch.device('cpu')
    
    if cuda_aware:
        import cupy
        ctx = cupy.cuda.Device(device_ordinal)
    else:
        ctx = nullcontext()

    return use_cuda, cuda_aware, device_ordinal, device, ctx


def compute_distribution_info(P: Partition, shape: List[int]) -> Dict[str, Any]:
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

def create_root_partition(P: Partition) -> Partition:
    P_root_base = P.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition([1]*P.dim)
    return P_root

def create_standard_partitions(shape: List[int]) -> Tuple[Partition, Partition, Partition]:
    from mpi4py import MPI
    P_world = Partition(MPI.COMM_WORLD)
    P_x_base = P_world.create_partition_inclusive(np.arange(np.prod(shape)))
    P_x = P_x_base.create_cartesian_topology_partition(shape)
    P_root = create_root_partition(P_x)
    return P_world, P_x, P_root

def alphabet(n: int, as_array=False):
    array = [chr(i+97) for i in range(n)]
    if as_array: return array
    return ''.join(array)

def unit_guassian_normalize(x):
    mu = torch.mean(x, 0).unsqueeze(0)
    std = torch.std(x, 0).unsqueeze(0)
    out = (x-mu)/(std+1e-6)
    return out, mu, std

def unit_gaussian_denormalize(x, mu, std):
    return x*(std + 1e-6) + mu