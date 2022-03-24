from dataclasses import dataclass
from distdl.utilities.slicing import *
from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from torch import Tensor
from typing import Callable, List, Optional, Tuple

import distdl
import distdl.nn as dnn
import numpy as np
import torch
import torch.nn as nn

Partition = distdl.backend.backend.Partition

@dataclass
class GradientTestResult:
    converged: bool
    convergence: Tuple[List[float], List[float], List[float]]

    def __str__(self):
        return f'{self.converged}, {self.convergence[0]}, {self.convergence[1]}, {self.convergence[2]}'

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

def gradient_test(P: Partition,
                  F: Callable[[Tensor], Tensor],
                  global_shape: np.ndarray,
                  max_iter: int = 5,
                  device: torch.device = torch.device('cpu'),
                  dtype: torch.dtype = torch.float32) -> Optional[GradientTestResult]:

    P_0_base = P.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1]*P.dim) 

    S = dnn.DistributedTranspose(P_0, P)
    G = dnn.DistributedTranspose(P, P_0)
    B = dnn.Broadcast(P_0, P)

    criterion = dnn.DistributedMSELoss(P).to(device)

    info = compute_distribution_info(P, global_shape)
    local_shape = info['shape']

    x = torch.rand(*local_shape, device=device, dtype=dtype)
    x0 = x + (1**(-P.dim))*torch.rand(*local_shape, device=device, dtype=dtype)
    dx = x-x0

    x.requires_grad = True
    x0.requires_grad = True

    y = F(x)
    y0 = F(x0)
    dy = y-y0
    y.backward(dy)
    
    f0 = criterion(y, y0).item()

    x_grad = x.grad.detach()
    x = x.detach()
    x0 = x0.detach()
    dx = dx.detach()

    h = zero_volume_tensor(device=device)
    if P_0.active:
        h = torch.tensor([1e-3*f0], device=device)
        err1 = []
        err2 = []
        hs = []

    h = B(h)

    for j in range(max_iter):
        
        x_new = x0 + h*dx
        y_new = F(x_new)
        f = criterion(y, y_new).item()

        if P_0.active:
            err1.append(abs(f-f0))
            err2.append(abs(f - f0 + h*torch.inner(dx.flatten(), x_grad.flatten())).cpu().item())
            hs.append(h.cpu().item())
            h = h/(2*f0)

        else:
            h = zero_volume_tensor(device=device)

        h = B(h)

    if P_0.active:
        err1_converged = np.isclose((err1[-1]/(err1[0]/2**(max_iter-1))), f0, atol=f0)
        err2_converged = np.isclose((err2[-1]/(err2[0]/4**(max_iter-1))), f0, atol=f0)
        converged = err1_converged and err2_converged
        return GradientTestResult(converged, (err1, err2, hs))

    else:
        return None

if __name__ == '__main__':

    from argparse import ArgumentParser
    from dfno.utils import create_standard_partitions
    from dfno import DistributedFNONd

    import gc

    parser = ArgumentParser()
    parser.add_argument('--input-shape',     '-is',    type=int, nargs='+')
    parser.add_argument('--partition-shape', '-ps',    type=int, nargs='+')
    parser.add_argument('--max-iter',        '-mi',    type=int, default=5)
    parser.add_argument('--device',          '-d',     type=str, default='cpu')
    parser.add_argument('--dtype',           '-t',     type=torch.dtype, default=torch.float32)
    parser.add_argument('--num-gpus',        '-ngpu',  type=int, default=1)
    parser.add_argument('--output',          '-o',     type=str, default='grad.txt')

    args = parser.parse_args()
    P_world, P_x, P_0 = create_standard_partitions(args.partition_shape)
    
    F = DistributedFNONd(
            P_x=P_x,
            width=20,
            modes=[4]*(len(args.input_shape)-2),
            out_timesteps=10,
            decomposition_order=1,
            num_blocks=4,
            device=torch.device('cpu'),
            dtype=args.dtype,
            P_y=P_x
    )

    info = compute_distribution_info(P_x, args.input_shape)
    local_shape = info['shape']
    dummy = torch.rand(*local_shape, device=torch.device('cpu'), dtype=args.dtype)
    with torch.no_grad():
        _ = F(dummy)

    gc.collect()

    device = torch.device(f'cuda:{P_x.rank % args.num_gpus}') if args.device == 'cuda' else torch.device('cpu')
    F = F.to(device)

    result = gradient_test(P=P_x,
                           F=F,
                           global_shape=args.input_shape,
                           max_iter=args.max_iter,
                           device=device,
                           dtype=args.dtype)
    
    if P_0.active:
        with open(args.output, 'w') as f:
            f.write(str(result))
