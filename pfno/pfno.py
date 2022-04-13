from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from torch import Tensor
from typing import Any, List, Tuple, Dict

import distdl
import distdl.nn as dnn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

Partition = distdl.backend.backend.Partition

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

class BroadcastedLinear(nn.Module):

    def __init__(self, P_x, in_features, out_features, dim=-1, bias=True, device=torch.device('cpu'), dtype=torch.float32):

        super().__init__()

        self.P_x = P_x
        self.in_features = in_features
        self.out_features = out_features
        self.scale = 1/np.sqrt(in_features*out_features)
        self.bias = bias
        
        self.b_shape = [1]*P_x.dim
        self.b_shape[dim] = out_features

        self.P_root = create_root_partition(P_x)
        if self.P_root.active:
            self.W = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
            self.b = nn.Parameter(torch.zeros(*self.b_shape, device=device, dtype=dtype))
            torch.nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        else:
            self.W = nn.Parameter(zero_volume_tensor(device=device))
            self.b = nn.Parameter(zero_volume_tensor(device=device))

        self.W_bcast = dnn.Broadcast(self.P_root, P_x)
        self.b_bcast = dnn.Broadcast(self.P_root, P_x)

        x_chars = alphabet(P_x.dim, as_array=True)
        y_chars = alphabet(P_x.dim, as_array=True)
        x_chars[dim] = 'i'
        y_chars[dim] = 'o'
        w_chars = 'oi'
        self.eqn = f"{w_chars},{''.join(x_chars)}->{''.join(y_chars)}"

        self.dt_comm = 0

    def forward(self, x: Tensor) -> Tensor:
        self.dt_comm = 0
        
        t0 = time.time()
        W = self.W_bcast(self.W)
        b = self.b_bcast(self.b)
        t1 = time.time()
        self.dt_comm += (t1-t0)

        y = torch.einsum(self.eqn, W, x)
        if self.bias:
            y += b
        return y

class ParallelFNOBlock(nn.Module):

    def __init__(self, P_x, in_shape, modes, device=torch.device('cpu'), dtype=torch.float32):

        super().__init__()

        self.P_x = P_x
        self.in_shape = in_shape
        self.width = in_shape[1]
        self.modes = modes
        self.n = P_x.dim-2
        self.device = device
        self.dtype = dtype
        self.dtype_complex = torch.complex64 if dtype == torch.float32 else torch.complex128

        # Setup FFT partitions
        shape_m = P_x.shape.copy()
        shape_y = P_x.shape.copy()

        n0 = int(np.ceil(self.n/2))
        n1 = int(np.floor(self.n/2))
        shape_m[2+n0:] = 1
        shape_m[2:2+n1] *= P_x.shape[2+n0:]
        shape_y[2:2+n0] = 1
        shape_y[2+n0:] *= P_x.shape[2:2+n1]

        self.dim_m = np.arange(2+n0, P_x.dim)
        self.dim_y = np.arange(2, 2+n0)

        self.P_m = P_x.create_cartesian_topology_partition(shape_m)
        self.P_y = P_x.create_cartesian_topology_partition(shape_y)

        self.R1 = dnn.Repartition(self.P_x, self.P_m)
        self.R2 = dnn.Repartition(self.P_m, self.P_y)
        self.R3 = dnn.Repartition(self.P_y, self.P_m)
        self.R4 = dnn.Repartition(self.P_m, self.P_x)

        # Setup weights
        self.scale = 1/(self.width*self.width)

        def make_weight(shape):
            return nn.Parameter(self.scale*torch.rand(self.width, self.width, *shape, device=self.device, dtype=self.dtype_complex))

        def make_slice(bounds):
            sl = [slice(None, None, 1), slice(None, None, 1)]
            for a, b in bounds:
                sl.append(slice(a, b, 1))
            return sl

        # Weights are in the bottom half of an n-dimensional hypercube. Thus, we
        # can map an n-digit binary number to low/high modes in the spatial
        # dimensions and low modes in the time dimension
        self.weights = nn.ParameterList([])
        self.slices = []
        
        fft_shape = [*in_shape[:-1], in_shape[-1]//2]
        info = compute_distribution_info(self.P_y, fft_shape)
        for i in range(2**(self.n-1)):
            
            s = bin(i)[2:].zfill(self.n)
            bounds = []

            for j, digit in enumerate(s):
                dim = self.P_y.dim-j-1
                mode = modes[dim-2]
                start = info['start'][dim]
                stop = info['stop'][dim]
                dim_size = fft_shape[dim]
                if digit == '0':
                    bounds.append((max(0, start)-start, min(mode, stop)-start))
                else:
                    bounds.append((max(dim_size-mode, start)-start, min(dim_size, stop)-start))
            
            bounds = list(reversed(bounds))
            valid = True
            for a, b in bounds:
                if b-a <= 0:
                    valid = False

            if valid:
                self.weights.append(make_weight([b-a for a, b in bounds]))
                self.slices.append(make_slice(bounds))

        # Setup einsum equation for spectral conv
        self.w_chars = alphabet(P_x.dim, as_array=True)
        self.x_chars = alphabet(P_x.dim, as_array=True)
        self.y_chars = alphabet(P_x.dim, as_array=True)
        self.w_chars[0] = 'i'
        self.w_chars[1] = 'o'
        self.x_chars[1] = 'i'
        self.y_chars[1] = 'o'
        self.eqn = f"{''.join(self.x_chars)},{''.join(self.w_chars)}->{''.join(self.y_chars)}"

        # Linear pass-through layer
        self.linear = BroadcastedLinear(self.P_x, self.width, self.width, bias=False, dim=1, device=device, dtype=dtype)

        self.dt_comm = 0

    def forward(self, x: Tensor) -> Tensor:
        self.dt_comm = 0

        y0 = self.linear(x)
        
        t0 = time.time()
        x = self.R1(x)
        self.dt_comm += (time.time()-t0)

        x = torch.fft.rfftn(x, dim=tuple(self.dim_m))

        t0 = time.time()
        x = self.R2(x)
        self.dt_comm += (time.time()-t0)

        x = torch.fft.fftn(x, dim=tuple(self.dim_y))

        y = 0*x.clone()
        for w, sl in zip(self.weights, self.slices):
            y[sl] = torch.einsum(self.eqn, x[sl], w)

        y = torch.fft.ifftn(y, dim=tuple(self.dim_y))

        t0 = time.time()
        y = self.R3(y)
        self.dt_comm += (time.time()-t0)

        y = torch.fft.irfftn(y, dim=tuple(self.dim_m))

        t0 = time.time()
        y = self.R4(y)
        self.dt_comm += (time.time()-t0)

        return F.gelu(y0 + y)

class ParallelFNO(nn.Module):

    def __init__(self, P_x, in_shape, out_timesteps, width, modes, num_blocks=4, device=torch.device('cpu'), dtype=torch.float32):

        super().__init__()

        self.P_x = P_x
        self.in_shape = in_shape
        self.out_timesteps = out_timesteps
        self.width = width
        self.modes = modes
        self.num_blocks = num_blocks
        self.device = device
        self.dtype = dtype

        self.block_in_shape = [in_shape[0], width, *in_shape[2:-1], out_timesteps]

        self.linear1 = BroadcastedLinear(P_x, in_shape[-1], out_timesteps, dim=-1, device=device, dtype=dtype)
        self.linear2 = BroadcastedLinear(P_x, in_shape[1],  width, dim=1, device=device, dtype=dtype)
        self.linear3 = BroadcastedLinear(P_x, width, 128, dim=1, device=device, dtype=dtype)
        self.linear4 = BroadcastedLinear(P_x, 128, 1, dim=1, device=device, dtype=dtype)

        self.blocks = nn.ModuleList([
            ParallelFNOBlock(
                self.P_x,
                self.block_in_shape,
                self.modes, 
                device=device,
                dtype=dtype
            ) for _ in range(num_blocks)
        ])

        self.bn1 = dnn.DistributedBatchNorm(P_x, self.width)
        self.bn2 = dnn.DistributedBatchNorm(P_x, self.width)

        self.dt_comm = 0

    def forward(self, x: Tensor) -> Tensor:
        self.dt_comm = 0

        x = self.linear1(x)
        self.dt_comm += self.linear1.dt_comm
        x = F.gelu(x)
        x = self.linear2(x)
        self.dt_comm += self.linear2.dt_comm
        x = F.gelu(x)

        #x = self.bn1(x)

        for block in self.blocks:
            x = block(x)
            self.dt_comm += block.dt_comm

        #x = self.bn2(x)

        x = self.linear3(x)
        self.dt_comm += self.linear3.dt_comm
        x = F.gelu(x)
        x = self.linear4(x)
        self.dt_comm += self.linear2.dt_comm
        return x

if __name__ == '__main__':
    
    P_shape = (1, 1, 2, 2, 1, 1)
    P_world, P_x, P_root = create_standard_partitions(P_shape)
    
    num_gpus = 1
    device_ordinal = f'{P_x.rank % num_gpus}'

    try:
        device = torch.device(f'cuda:{device_ordinal}')
        import cupy
        ctx = cupy.cuda.Device(device_ordinal)
    except:
        device = torch.device('cpu')
        from contextlib import nullcontext
        ctx = nullcontext()

    width = 20
    modes = (4, 4, 4, 8)
    nt = 30
    shape = (64, 64, 64)
    local_shape = (32, 32, 64)
    x = torch.rand(1, 1, *local_shape, 1, device=device, dtype=torch.float32)

    in_shape = (1, 1, *shape, 1)
    with ctx:
        network = ParallelFNO(P_x, in_shape, nt, width, modes, num_blocks=4, device=x.device, dtype=x.dtype)
        criterion = dnn.DistributedMSELoss(P_x)
        y = network(x)
        
        for i in range(10):
            t0 = time.time()
            y = network(x)
            t1 = time.time()
            print(f'rank = {P_x.rank}, dt = {t1-t0}')

            loss = criterion(y, torch.rand_like(y))
            P_x._comm.Barrier()
            
            t0 = time.time()
            loss.backward()
            t1 = time.time()
            print(f'rank = {P_x.rank}, dt_grad = {t1-t0}')
