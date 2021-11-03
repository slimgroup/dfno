from collections import OrderedDict
import distdl
import distdl.nn as dnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from dfno.utils import create_root_partition

class DXFFTN(nn.Module):

    def __init__(self, P_x, info, P_y=None):
        super(DXFFTN, self).__init__()

        self.P_x = P_x
        self.info = info
        
        self.partitions = [P_x]
        self.transposes = nn.ModuleList([])
        self.transforms = []
        self.transform_kwargs = []

        for dims, data in info.items():
            shape = P_x.shape.copy()
            for d, pd in zip(dims, data['prod_dims']):
                shape[d] = 1
                shape[pd] *= P_x.shape[d]
            self.partitions.append(P_x.create_cartesian_topology_partition(shape))
            self.transposes.append(dnn.DistributedTranspose(self.partitions[-2], self.partitions[-1]))
            self.transforms.append(data['transform'])
            self.transform_kwargs.append(data['transform_kwargs'])

        self.P_y = self.partitions[-1] if P_y is None else P_y
        self.transpose_out = dnn.DistributedTranspose(self.partitions[-1], self.P_y)

    def forward(self, x):
        for T, f, kwargs in zip(self.transposes, self.transforms, self.transform_kwargs):
            x = T(x)
            x = f(x, **kwargs)
        x = self.transpose_out(x)
        return x

class DistributedSpectralConvNd(nn.Module):

    def __init__(self,
                 P_x,
                 out_channels,
                 modes,
                 decomposition_order=1,
                 device=None,
                 dtype=None,
                 P_y=None):

        super(DistributedSpectralConvNd, self).__init__()

        self.P_x = P_x
        self.in_channels = None
        self.out_channels = out_channels
        self.modes = modes
        self.decomposition_order = decomposition_order
        self.device = device
        self.dtype = dtype
        self.P_y = P_x if P_y is None else P_y

        self.dim = P_x.dim
        self.flag_exp = 2**(self.dim-2)

        info = []
        for i in range(2, self.dim, decomposition_order):
            start = i
            stop = min(i+decomposition_order, self.dim)
            dims = tuple(range(start, stop))
            transform = torch.fft.rfftn if stop == self.dim else torch.fft.fftn
            transform_kwargs = {'dim': dims}
            prod_dims = tuple((d+decomposition_order)%self.dim + 2 for d in dims) if stop == self.dim \
                else tuple(min(d+decomposition_order, self.dim-1) for d in dims)
            info.append((dims, {'transform': transform, 'transform_kwargs': transform_kwargs, 'prod_dims': prod_dims}))
        
        self.drfftn_info = OrderedDict(reversed(info.copy()))
        self.drfftn = DXFFTN(P_x, self.drfftn_info)
        self.P_ft = self.drfftn.P_y

        for k, v in info:
            t = v['transform']
            v['transform'] = torch.fft.irfftn if t == torch.fft.rfftn else torch.fft.ifftn
        self.dirfftn_info = OrderedDict(info)
        self.dirfftn = DXFFTN(self.P_ft, self.dirfftn_info, P_y=self.P_y)

        self.slices = [None for _ in range(0, self.flag_exp, 2)]
        self.weights = nn.ParameterList([nn.UninitializedParameter(device=device, dtype=dtype) for _ in range(0, self.flag_exp, 2)])
        self.use_weights = [False for _ in range(0, self.flag_exp, 2)]
        self.is_init = False

        chars = ''.join(chr(i+97) for i in range(self.dim-2))
        self.eqn = f'xy{chars},yz{chars}->xz{chars}'

    def forward(self, x):

        x_ft = self.drfftn(x)

        with torch.no_grad():
            if not self.is_init:
                self.in_channels = x.shape[1]
                self.scale = 1/(self.in_channels*self.out_channels)

                x_ft_ls = TensorStructure(x_ft)
                x_ft_gs = distdl.backend.backend.tensor_comm.assemble_global_tensor_structure(x_ft_ls, self.P_ft)
                x_ft_shapes = compute_subtensor_shapes_balanced(x_ft_gs, self.P_ft.shape)
                x_ft_starts = compute_subtensor_start_indices(x_ft_shapes)
                x_ft_stops = compute_subtensor_stop_indices(x_ft_shapes)
                x_ft_index = tuple(self.P_ft.index)
                x_ft_shape = x_ft_shapes[x_ft_index]
                x_ft_start = x_ft_starts[x_ft_index]
                x_ft_stop = x_ft_stops[x_ft_index]

                modes_low_stops = np.minimum(self.modes, x_ft_stop[2:]) - x_ft_start[2:]
                modes_high_starts = np.maximum(x_ft_gs.shape[2:]-self.modes, x_ft_start[2:]) - x_ft_start[2:]

                for i in range(0, self.flag_exp, 2):
                    s = bin(i)[2:].zfill(self.dim-2)
                    flags = [int(c) for c in s]
                    slices = [slice(None, None, 1), slice(None, None, 1)]
                    shape = [self.in_channels, self.out_channels]

                    for j, f in enumerate(flags):
                        start = 0 if f == 0 else modes_high_starts[j]
                        stop = modes_low_stops[j] if f == 0 else x_ft_shape[j+2]
                        slices.append(slice(start, stop, 1))
                        shape.append(stop-start)

                    j = i//2
                    shape = np.array(shape)
                    if (shape > 0).all():
                        self.slices[j] = tuple(slices)
                        self.weights[j].materialize(tuple(shape))
                        nn.init.uniform_(self.weights[j], a=0.0, b=self.scale)
                        self.use_weights[j] = True
                    else:
                        self.weights[j].materialize([1]*self.dim)
                        self.use_weights[j] = False

                self.is_init = True

        out_ft = x_ft.clone()
        out_ft[:] = 0.0

        for sl, w, uw in zip(self.slices, self.weights, self.use_weights):
            if uw:
                out_ft[sl] = torch.einsum(self.eqn, x_ft[sl], w)

        x_out = self.dirfftn(out_ft)
        return x_out

class BroadcastedAffineOperator(nn.Module):

    def __init__(self, P_x, out_features, contraction_dim, bias=True, device=None, dtype=None, P_y=None):
        super(BroadcastedAffineOperator, self).__init__()

        self.P_x = P_x
        self.in_features = None
        self.out_features = out_features
        self.contraction_dim = contraction_dim
        self.bias = bias
        self.device = device
        self.dtype = dtype
        self.P_y = P_x if P_y is None else P_y

        self.dim = P_x.dim
        self.P_0 = create_root_partition(P_x)

        self.W_shape = (None, out_features)
        self.b_shape = [1]*self.dim
        self.b_shape[contraction_dim] = self.out_features

        self.W_scale = None
        self.b_scale = 1/out_features

        self.W_bc = dnn.Broadcast(self.P_0, self.P_x)
        self.b_bc = dnn.Broadcast(self.P_0, self.P_x) if bias else None

        if self.P_0.active:
            self.W = nn.UninitializedParameter(device=device, dtype=dtype)
            self.b = nn.Parameter(self.b_scale * torch.rand(*self.b_shape, device=device, dtype=dtype)) if bias else None
        else:
            self.W = nn.Parameter(zero_volume_tensor(device=device))
            self.b = nn.Parameter(zero_volume_tensor(device=device)) if bias else None

        self.is_init = False

        self.T_out = dnn.DistributedTranspose(self.P_x, self.P_y)

        chars = [chr(i+97) for i in range(self.dim)]
        chars[contraction_dim] = 'x'
        in_chars = ''.join(chars)
        chars = [chr(i+97) for i in range(self.dim)]
        chars[contraction_dim] = 'y'
        out_chars = ''.join(chars)
        self.eqn = f'{in_chars},xy->{out_chars}'

    def forward(self, x):
        with torch.no_grad():
            if not self.is_init:
                self.in_features = x.shape[self.contraction_dim]
                self.W_shape = (self.in_features, self.out_features)
                self.W_scale = 1/(self.in_features*self.out_features)
                if self.P_0.active:
                    self.W.materialize(self.W_shape, device=self.device, dtype=self.dtype)
                    nn.init.uniform_(self.W, a=0.0, b=self.W_scale)
                self.is_init = True

        W = self.W_bc(self.W)
        x = torch.einsum(self.eqn, x, W)
        if self.bias:
            b = self.b_bc(self.b)
            x = x + b
        x = self.T_out(x)
        return x

class DistributedFNONd(nn.Module):

    def __init__(self, P_x, width, modes, out_timesteps, decomposition_order=1, num_blocks=4, device=None, dtype=None, P_y=None):
        super(DistributedFNONd, self).__init__()

        self.P_x = P_x
        self.width = width
        self.modes = modes
        self.out_timesteps = out_timesteps
        self.decomposition_order = decomposition_order
        self.num_blocks = num_blocks
        self.device = device
        self.dtype = dtype
        self.sconv_dtype = torch.cdouble if dtype == torch.float64 else torch.cfloat
        self.P_y = P_x if P_y is None else P_y

        self.dim = P_x.dim
        shape = P_x.shape.copy()
        shape[-1] = 1
        shape[-2] *= P_x.shape[-1]
        self.P_t = P_x.create_cartesian_topology_partition(shape)

        self.fcs = nn.ModuleList([
            BroadcastedAffineOperator(self.P_x, width, 1, device=device, dtype=dtype, P_y=self.P_t),
            BroadcastedAffineOperator(self.P_t, out_timesteps, -1, device=device, dtype=dtype, P_y=P_x),
            BroadcastedAffineOperator(self.P_x, 128, 1, device=device, dtype=dtype),
            BroadcastedAffineOperator(self.P_x, 1, 1, device=device, dtype=dtype, P_y=self.P_y)
        ])

        self.sconvs = nn.ModuleList([
            DistributedSpectralConvNd(
                P_x,
                width,
                modes,
                decomposition_order=decomposition_order,
                device=device,
                dtype=self.sconv_dtype) for _ in range(num_blocks)
        ])

        self.affines = nn.ModuleList([
            BroadcastedAffineOperator(
                P_x,
                width,
                1,
                device=device,
                dtype=dtype
            ) for _ in range(num_blocks)
        ])

        self.bn0 = dnn.DistributedBatchNorm(P_x, width)
        self.bn1 = dnn.DistributedBatchNorm(P_x, width)
        self.T_out = dnn.DistributedTranspose(self.P_x, self.P_y)

    def forward(self, x):

        x = self.fcs[0](x)
        x = F.gelu(x)
        x = self.fcs[1](x)
        x = F.gelu(x)

        #x = self.bn0(x)

        for S, A in zip(self.sconvs, self.affines):
            x1 = S(x)
            x2 = A(x)
            x = x1 + x2
            x = F.gelu(x)

        #x = self.bn1(x)

        x = self.fcs[2](x)
        x = F.gelu(x)
        x = self.fcs[3](x)

        x = self.T_out(x)
        return x
