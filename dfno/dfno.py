from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from torch import Tensor
from .utils import alphabet, create_root_partition, compute_distribution_info

import copy
import distdl
import distdl.nn as dnn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

Partition = distdl.backend.backend.Partition

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

class DistributedFNOBlock(nn.Module):

    def __init__(self, P_x, in_shape, modes, fft_order=None, device=torch.device('cpu'), dtype=torch.float32):

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

        if fft_order is not None:
            before, after = fft_order

            # sanity check the given dimensions (must be 2..N+2)
            given_dims = set(before+after)
            # can't contain 0 or 1
            if 0 in given_dims:
                raise Exception("dimension 0 (batch element) cannot be part of the FFT ordering")
            if 1 in given_dims:
                raise Exception("dimension 1 (channel) cannot be part of the FFT ordering")
            # before and after can't overlap
            if len(before+after) != len(given_dims):
                dims_before_after = sorted([0,1] + before + after)
                raise Exception(f"fft-order-before ({before}) overlaps with fft-order-after ({after})!")
            # all dimensions must be present
            missing_dims = set(range(2,self.n+2))
            for dim in given_dims:
                missing_dims.discard(dim)
            if len(missing_dims) > 0:
                raise Exception(f"FFT order must include dimensions {missing_dims}")
            # reject unknown extra dimensions
            extra_dims = given_dims.copy()
            for dim in range(2,P_x.dim):
                extra_dims.discard(dim)
            if len(extra_dims) > 0:
                raise Exception(f"FFT order includes unknown extra dimensions {extra_dims}")

            self.dim_m = np.array(before, dtype=int)
            self.dim_y = np.array(after , dtype=int)

            if len(self.dim_m) == len(self.dim_y):
                # equal split before and after
                shape_m[self.dim_m] = 1
                shape_y[self.dim_y] = 1
                shape_m[self.dim_y] *= P_x.shape[self.dim_m]
                shape_y[self.dim_m] *= P_x.shape[self.dim_y]
            elif len(self.dim_y) > 1:
                # more than 1 FFT after repartition
                # TODO: figure out how to calculate a shape for this
                raise Exception("TODO: implement uneven splits with more than one inner dimension")
            else:
                # N-1 FFTs before repartition, 1 FFT after
                if np.prod(shape_y[self.dim_y]) != 1:
                    raise Exception("Please set the --partition_shape sizes of --fft-order-after dimensions to 1.")

                shape_m[self.dim_m] = 1
                shape_m[self.dim_y] = np.prod(shape_y[self.dim_m])
                # leave shape_y alone

        else:
            n0 = int(np.ceil(self.n/2))
            n1 = int(np.floor(self.n/2))
            shape_m[2+n0:] = 1
            shape_m[2:2+n1] *= P_x.shape[2+n0:]
            shape_y[2:2+n0] = 1
            shape_y[2+n0:] *= P_x.shape[2:2+n1]

            self.dim_m = np.arange(2+n0, P_x.dim)
            self.dim_y = np.arange(2, 2+n0)

        print(f"DistributedFNOBlock: P_x.shape: {P_x.shape} dim_m: {self.dim_m} dim_y: {self.dim_y}  shape_m: {shape_m} shape_y: {shape_y}")

        self.P_m = P_x.create_cartesian_topology_partition(shape_m)
        self.P_y = P_x.create_cartesian_topology_partition(shape_y)

        self.R1 = dnn.Repartition(self.P_x, self.P_m)
        self.R2 = dnn.Repartition(self.P_m, self.P_y)
        self.R3 = dnn.Repartition(self.P_y, self.P_m)
        self.R4 = dnn.Repartition(self.P_m, self.P_x)

        # Setup FFT restrictions
        self.restrict_prefixes = {}
        self.restrict_suffixes = {}
        for dim in [*self.dim_m, *self.dim_y]:
            mode = modes[dim-2]
            self.restrict_prefixes[dim] = mode
            if dim != self.dim_m[-1]:
                self.restrict_suffixes[dim] = mode

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
        for dim, restriction in self.restrict_prefixes.items():
            fft_shape[dim] = restriction
        for dim, restriction in self.restrict_suffixes.items():
            fft_shape[dim] += restriction
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

    def restrict(self, x: Tensor, dim: int) -> Tensor:
        '''Discard unused higher-frequency elements.'''
        if dim not in self.restrict_prefixes and dim not in self.restrict_suffixes:
            # nothing to restrict.
            return y

        pieces = []
        sl = [slice(None,None,1)] * len(x.shape)

        if dim in self.restrict_prefixes:
            # add the prefix block
            sl[dim] = slice(None, self.restrict_prefixes[dim], 1)
            pieces.append(x[sl])

        if dim in self.restrict_suffixes:
            # add the suffix block
            sl[dim] = slice(-self.restrict_suffixes[dim], None, 1)
            pieces.append(x[sl])

        if len(pieces) == 1:
            # only keeping a single piece
            return pieces[0]

        # multiple pieces, concatenate them
        x = torch.cat(pieces, dim=dim)
        return x

    def zeropad(self, y: Tensor, dim: int, target_shape: list) -> Tensor:
        '''Fill in zeroes for higher-frequency elements.'''

        if dim not in self.restrict_prefixes and dim not in self.restrict_suffixes:
            # nothing was restricted; nothing to zero-pad.
            return y

        # pad up to the target shape
        pad_shape = copy.copy(target_shape)
        pad_shape[dim] -= y.shape[dim]
        for i in pad_shape:
            if i < 1:
                # pad is empty
                return y

        # build an array of pieces: the prefix if any, then zeroes, then suffix if any
        pieces = []
        sl = [slice(None,None,1)] * len(y.shape)

        if dim in self.restrict_prefixes:
            # add the prefix block
            sl[dim] = slice(None, self.restrict_prefixes[dim], 1)
            pieces.append(y[sl])

        pieces.append(torch.zeros(pad_shape, dtype=y.dtype, layout=y.layout, device=y.device))

        if dim in self.restrict_suffixes:
            # add the suffix block
            sl[dim] = slice(-self.restrict_suffixes[dim], None, 1)
            pieces.append(y[sl])

        # assemble the pieces
        y = torch.cat(pieces, dim=dim)

        return y

    def forward(self, x: Tensor) -> Tensor:
        self.dt_comm = 0

        y0 = self.linear(x)

        t0 = time.time()
        x = self.R1(x)
        self.dt_comm += (time.time()-t0)

        saved_shapes = {}
        outermost_dim = self.dim_m[-1]
        x = torch.fft.rfft(x, dim=outermost_dim)
        saved_shapes[outermost_dim] = list(x.shape)
        x = self.restrict(x, outermost_dim)
        for dim in reversed(self.dim_m[:-1]):
            x = torch.fft.fft(x, dim=dim)
            saved_shapes[dim] = list(x.shape)
            x = self.restrict(x, dim)

        t0 = time.time()
        x = self.R2(x)
        self.dt_comm += (time.time()-t0)

        for dim in reversed(self.dim_y):
            x = torch.fft.fft(x, dim=dim)
            saved_shapes[dim] = list(x.shape)
            x = self.restrict(x, dim)

        y = 0*x.clone()
        for w, sl in zip(self.weights, self.slices):
            y[sl] = torch.einsum(self.eqn, x[sl], w)

        for dim in self.dim_y:
            y = self.zeropad(y, dim, saved_shapes[dim])
            y = torch.fft.ifft(y, dim=dim)

        t0 = time.time()
        y = self.R3(y)
        self.dt_comm += (time.time()-t0)

        for dim in self.dim_m[:-1]:
            y = self.zeropad(y, dim, saved_shapes[dim])
            y = torch.fft.ifft(y, dim=dim)
        y = self.zeropad(y, outermost_dim, saved_shapes[outermost_dim])
        y = torch.fft.irfft(y, dim=outermost_dim)

        t0 = time.time()
        y = self.R4(y)
        self.dt_comm += (time.time()-t0)

        return F.gelu(y0 + y)

class DistributedFNO(nn.Module):

    def __init__(self, P_x, in_shape, out_timesteps, width, modes, num_blocks=4, fft_order=None, device=torch.device('cpu'), dtype=torch.float32):

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
            DistributedFNOBlock(
                self.P_x,
                self.block_in_shape,
                self.modes,
                device=device,
                dtype=dtype,
                fft_order=fft_order
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

    from utils import get_env, create_standard_partitions

    P_shape = (1, 1, 2, 2, 1, 1)
    P_world, P_x, P_root = create_standard_partitions(P_shape)
    num_gpus = 1
    use_cuda, cuda_aware, device_ordinal, device, ctx = get_env(P_x, num_gpus=num_gpus)

    width = 20
    modes = (4, 4, 4, 8)
    nt = 30
    shape = (64, 64, 64)
    local_shape = (32, 32, 64)
    x = torch.rand(1, 1, *local_shape, 1, device=device, dtype=torch.float32)

    in_shape = (1, 1, *shape, 1)
    with ctx:
        network = DistributedFNO(P_x, in_shape, nt, width, modes, num_blocks=4, device=x.device, dtype=x.dtype)
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
