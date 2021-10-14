import distdl
import distdl.nn as dnn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *

class DXFFTN(nn.Module):

    def __init__(self, P_x, info, P_y=None):
        
        super(DXFFTN, self).__init__()
        
        self.P_x = P_x
        self.info = info
        
        self.transforms = []
        self.transform_kwargs = []
        self.partitions = [P_x]
        self.transposes = []
        
        for dims, data in info.items():
            self.transforms.append(data['transform'])
            self.transform_kwargs.append(data['transform_kwargs'])

            shape = P_x.shape.copy()
            for d, pd in zip(dims, data['prod_dims']):
                shape[d] = 1
                shape[pd] *= P_x.shape[d]

            self.partitions.append(P_x.create_cartesian_topology_partition(shape))
            self.transposes.append(dnn.DistributedTranspose(self.partitions[-2], self.partitions[-1]))

        self.P_y = self.partitions[-1] if P_y is None else P_y
        self.T_out = dnn.DistributedTranspose(self.partitions[-1], self.P_y)

    def forward(self, x):

        for T, f, kwargs in zip(self.transposes, self.transforms, self.transform_kwargs):
            x = T(x)
            x = f(x, **kwargs)

        x = self.T_out(x)
        return x

class DistributedSpectralConvNd(nn.Module):

    def __init__(self, P_x, in_channels, out_channels, modes, decomposition_order=1, P_y=None):
        
        super(DistributedSpectralConvNd, self).__init__()

        self.P_x = P_x
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.decomposition_order = decomposition_order

        self.dim = P_x.dim
        self.scale = 1 / (in_channels*out_channels)

        info = []
        for i in range(2, self.dim, decomposition_order):
            start = i
            stop = min(i+decomposition_order, self.dim)
            dims = tuple(range(start, stop))
            
            if stop == self.dim:
                transform = torch.fft.rfftn
                transform_kwargs = {'dim': dims}
                prod_dims = tuple((d+decomposition_order)%self.dim+2 for d in dims)

            else:
                transform = torch.fft.fftn
                transform_kwargs = {'dim': dims}
                prod_dims = tuple(min(d+decomposition_order, self.dim-1) for d in dims)

            data = {'transform': transform, 'transform_kwargs': transform_kwargs, 'prod_dims': prod_dims}
            info.append((dims, data))

        info = OrderedDict(reversed(info))
        self.drfftn = DXFFTN(P_x, info)

        self.P_ft = self.drfftn.P_y

        info = []
        self.dirfftn_dims = []
        for i in range(2, self.dim, decomposition_order):
            start = i
            stop = min(i+decomposition_order, self.dim)
            dims = tuple(range(start, stop))
            self.dirfftn_dims.append(dims)
            
            if stop == self.dim:
                transform = torch.fft.irfftn
                transform_kwargs = {'dim': dims}
                prod_dims = tuple((d+decomposition_order)%self.dim+2 for d in dims)

            else:
                transform = torch.fft.ifftn
                transform_kwargs = {'dim': dims}
                prod_dims = tuple(min(d+decomposition_order, self.dim-1) for d in dims)

            data = {'transform': transform, 'transform_kwargs': transform_kwargs, 'prod_dims': prod_dims}
            info.append((dims, data))

        info = OrderedDict(info)
        self.dirfftn = DXFFTN(self.P_ft, info, P_y=P_y)
        self.P_y = self.dirfftn.partitions[-1]

        self.flag_exp = 2**(self.dim-2)
        self.slices = [None for _ in range(0, self.flag_exp, 2)]
        self.weights = nn.ParameterList([nn.UninitializedParameter() for _ in range(0, self.flag_exp, 2)])
        self.use_weights = [False for _ in range(0, self.flag_exp, 2)]

        chars = ''.join(chr(i+97) for i in range(self.dim-2))
        self.eqn = f'xy{chars},yz{chars}->xz{chars}'

        self.is_init = False
        self.pass_through = True

    def forward(self, x):

        x_ft = self.drfftn(x)

        with torch.no_grad():
            if not self.is_init:
                x_ls = TensorStructure(x)
                x_gs = distdl.backend.backend.tensor_comm.assemble_global_tensor_structure(x_ls, self.P_x)
                r_dims = tuple(np.atleast_1d(self.dirfftn_dims[-1]))
                self.dirfftn.transform_kwargs[-1]['s'] = tuple(np.atleast_1d(x_gs.shape[r_dims]))

                x_ft_ls = TensorStructure(x_ft)
                x_ft_gs = distdl.backend.backend.tensor_comm.assemble_global_tensor_structure(x_ft_ls, self.P_ft)

                x_ft_shapes = compute_subtensor_shapes_balanced(x_ft_gs, self.P_ft.shape)
                x_ft_starts = compute_subtensor_start_indices(x_ft_shapes)
                x_ft_stops = compute_subtensor_stop_indices(x_ft_shapes)

                index = tuple(self.P_ft.index)
                x_ft_shape = x_ft_shapes[index]
                x_ft_start = x_ft_starts[index]
                x_ft_stop = x_ft_stops[index]

                modes_low_stops = np.minimum(self.modes, x_ft_stop[2:]) - x_ft_start[2:]
                modes_high_starts = np.maximum(x_ft_gs.shape[2:]-self.modes, x_ft_start[2:]) - x_ft_start[2:]

                for j, i in enumerate(range(0, self.flag_exp, 2)):
                    s = bin(i)[2:].zfill(self.dim-2)
                    flags = [int(c) for c in s]

                    slices = [slice(None, None, 1), slice(None, None, 1)]
                    shape = [self.in_channels, self.out_channels]

                    for k, f in enumerate(flags):
                        start = 0 if f == 0 else modes_high_starts[k]
                        stop = modes_low_stops[k] if f == 0 else x_ft_shape[2+k]
                        shape.append(stop-start)
                        slices.append(slice(start, stop, 1))
                    
                    shape = np.array(shape)
                    if (shape > 0).all():
                        self.slices[j] = tuple(slices)
                        self.weights[j].materialize(shape=tuple(shape), dtype=torch.cfloat, device=x_ft.device)
                        self.weights[j][:] = self.scale*torch.rand(*shape, dtype=torch.cfloat, device=x_ft.device)
                        self.use_weights[j] = True
                        self.pass_through = False

                    else:
                        self.weights[j].materialize(shape=(1, 1))
                        self.weights[j][:] = 0
                        self.use_weights[j] = False

                self.is_init = True

        out_ft = x_ft.clone()
        out_ft[:] = 0.0
        
        if not self.pass_through:
            for sl, w, u in zip(self.slices, self.weights, self.use_weights):
                if u:
                    out_ft[sl] = torch.einsum(self.eqn, x_ft[sl], w)

        x = self.dirfftn(out_ft)
        return x

class DistributedFNONd(nn.Module):

    def __init__(self, P_x, width, modes, out_timesteps, decomposition_order=1, num_blocks=4, P_y=None):
        
        super(DistributedFNONd, self).__init__()

        self.P_x = P_x
        self.width = width
        self.modes = modes
        self.out_timesteps = out_timesteps
        self.decomposition_order = decomposition_order
        self.num_blocks = num_blocks
        self.P_y = P_x if P_y is None else P_y

        self.dim = P_x.dim

        self.P_0_base = P_x.create_partition_inclusive([0])
        self.P_0 = self.P_0_base.create_cartesian_topology_partition([1]*self.dim)

        shape = P_x.shape.copy()
        shape[-1] = 1
        shape[-2] *= P_x.shape[-1]
        self.P_t = P_x.create_cartesian_topology_partition(shape)

        self.convs = nn.ModuleList([DistributedSpectralConvNd(P_x, width, width, modes, decomposition_order, P_y=P_x) for _ in range(num_blocks)])
        
        if self.P_0.active:
            self.fcs = nn.ParameterList([
                    nn.UninitializedParameter(),
                    nn.UninitializedParameter(),
                    nn.Parameter(torch.rand(width, 128)),
                    nn.Parameter(torch.rand(128, 1))
            ])
            self.weights = nn.ParameterList([nn.Parameter(torch.rand(width, width)) for _ in range(num_blocks)])
            self.biases = nn.ParameterList([nn.Parameter(torch.rand(1, width, *[1]*(self.dim-2))) for _ in range(num_blocks)])
        
        else:
            self.fcs = [nn.Parameter(zero_volume_tensor()) for _ in range(4)]
            self.weights = [nn.Parameter(zero_volume_tensor()) for _ in range(num_blocks)]
            self.biases = [nn.Parameter(zero_volume_tensor()) for _ in range(num_blocks)]

        self.fc_bcs = nn.ModuleList([
                dnn.Broadcast(self.P_0, self.P_x),
                dnn.Broadcast(self.P_0, self.P_t),
                dnn.Broadcast(self.P_0, self.P_y),
                dnn.Broadcast(self.P_0, self.P_y),
        ])
        self.weight_bcs = nn.ModuleList([dnn.Broadcast(self.P_0, self.P_x) for _ in range(num_blocks)])
        self.bias_bcs = nn.ModuleList([dnn.Broadcast(self.P_0, self.P_x) for _ in range(num_blocks)])

        self.T0 = dnn.DistributedTranspose(self.P_x, self.P_t)
        self.T1 = dnn.DistributedTranspose(self.P_t, self.P_x)
        self.T_out = dnn.DistributedTranspose(self.P_x, self.P_y)

        chars = ''.join(chr(i+97) for i in range(self.dim-2))
        self.channel_eqn = f'xy{chars},yz->xz{chars}'
        self.time_eqn = f'x{chars}y,yz->x{chars}z'

        self.is_init = False

    def forward(self, x):

        with torch.no_grad():
            if not self.is_init:
                x_ls = TensorStructure(x)
                x_gs = distdl.backend.backend.assemble_global_tensor_structure(x_ls, self.P_x)
                
                if self.P_0.active:
                    self.fcs[0].materialize(shape=(x.shape[1], self.width))
                    self.fcs[0][:] = torch.rand(x.shape[1], self.width)
                    self.fcs[1].materialize(shape=(x_gs.shape[-1], self.out_timesteps))
                    self.fcs[1][:] = torch.rand(x_gs.shape[-1], self.out_timesteps)

                self.is_init = True

        fc0 = self.fc_bcs[0](self.fcs[0])
        x = torch.einsum(self.channel_eqn, x, fc0)
        x = F.gelu(x)

        fc1 = self.fc_bcs[1](self.fcs[1])
        x = self.T0(x)
        x = torch.einsum(self.time_eqn, x, fc1)
        x = self.T1(x)
        x = F.gelu(x)

        for S, w, b, B1, B2 in zip(self.convs, self.weights, self.biases, self.weight_bcs, self.bias_bcs):
            w_bc = B1(w)
            b_bc = B2(b)
            x1 = S(x)
            x2 = torch.einsum(self.channel_eqn, x, w_bc) + b_bc
            x = x1 + x2
            x = F.gelu(x)

        fc2 = self.fc_bcs[2](self.fcs[2])
        x = torch.einsum(self.channel_eqn, x, fc2)
        x = F.gelu(x)

        fc3 = self.fc_bcs[3](self.fcs[3])
        x = torch.einsum(self.channel_eqn, x, fc3)

        return x 
