import distdl
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from dxfftn import DXFFTN
from torch import Tensor
from typing import List, Optional

Partition = distdl.backend.backend.Partition

class DistributedSpectralConvNd(nn.Module):

    def __init__(self,
                 P_x: Partition,
                 in_channels: int,
                 out_channels: int,
                 modes: List[int],
                 P_y: Optional[Partition] = None,
                 decomposition_order: int = 2) -> None:

        super(DistributedSpectralConvNd, self).__init__()

        self.P_x = P_x
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.P_y = P_y
        self.decomposition_order = decomposition_order

        self.scale = 1 / (in_channels * out_channels)
        self.dim = self.P_x.dim

        self.drfftn_info = OrderedDict()

        for i in reversed(range(2, self.dim, self.decomposition_order)):
            start = i
            stop = min(i+self.decomposition_order, self.dim)

            dims = tuple(range(start, stop))
            transform = torch.fft.rfftn if stop == self.dim else torch.fft.fftn
            transform_kwargs = {'dim': dims}
            repartition_dims = tuple(min(d+self.decomposition_order, self.dim-1) for d in dims) if stop < self.dim \
                else tuple((d+self.decomposition_order+2) % self.dim for d in dims)

            self.drfftn_info[dims] = {
                'transform': transform,
                'transform_kwargs': transform_kwargs,
                'repartition_dims': repartition_dims
            }

        self.drfftn = DXFFTN(self.P_x, self.drfftn_info)
        self.dirfftn = None

        self.P_ft = self.drfftn.P_y

        xi = 'xy'
        wi = 'yz'
        yi = 'xz'
        si = ''.join(chr(i+97) for i in range(self.dim-2))

        self.compl_mulnd_einsum_equation = f'{xi}{si},{wi}{si}->{yi}{si}'

    def compl_mulnd(self, x: Tensor, w: Tensor) -> Tensor:
        return torch.einsum(self.compl_mulnd_einsum_equation, x, w)

    def forward(self, x: Tensor) -> Tensor:

        if not self.P_x.active:
            return x

        with torch.no_grad():
            if self.dirfftn is None:

                    self.x_local_structure = TensorStructure(x)
                    self.x_global_structure = \
                        distdl.backend.backend.tensor_comm.assemble_global_tensor_structure(self.x_local_structure, self.P_x)

                    self.dirfftn_info = OrderedDict()

                    for i in reversed(range(2, self.dim, self.decomposition_order)):
                        start = i
                        stop = min(i+self.decomposition_order, self.dim)

                        dims = tuple(range(start, stop))
                        transform = torch.fft.irfftn if stop == self.dim else torch.fft.ifftn
                        transform_kwargs = {'dim': dims, 's': tuple(self.x_global_structure.shape[d] for d in dims)} if stop == self.dim \
                            else {'dim': dims}
                        repartition_dims = tuple(min(d+self.decomposition_order, self.dim-1) for d in dims) if stop < self.dim \
                            else tuple((d+self.decomposition_order+2) % self.dim for d in dims)

                        self.dirfftn_info[dims] = {
                            'transform': transform,
                            'transform_kwargs': transform_kwargs,
                            'repartition_dims': repartition_dims
                        }

                    self.dirfftn = DXFFTN(self.P_ft, self.dirfftn_info, P_y=self.P_y)
                    self.P_y = self.dirfftn.P_y

                    self.x_ft_global_structure = TensorStructure()
                    self.x_ft_global_structure.shape = np.copy(self.x_global_structure.shape)
                    self.x_ft_global_structure.shape[-1] = self.x_ft_global_structure.shape[-1] // 2 + 1

                    self.x_ft_shapes = compute_subtensor_shapes_balanced(self.x_ft_global_structure, self.P_ft.shape)
                    self.x_ft_starts = compute_subtensor_start_indices(self.x_ft_shapes)
                    self.x_ft_stops = compute_subtensor_stop_indices(self.x_ft_shapes)

                    self.P_ft_index = tuple(self.P_ft.index)
                    self.x_ft_shape = self.x_ft_shapes[self.P_ft_index]
                    self.x_ft_start = self.x_ft_starts[self.P_ft_index]
                    self.x_ft_stop = self.x_ft_stops[self.P_ft_index]

                    self.modes_low_stops = np.minimum(self.modes, self.x_ft_stop[2:]) - self.x_ft_start[2:]
                    self.modes_high_starts = np.maximum(self.x_ft_global_structure.shape[2:] - self.modes, self.x_ft_start[2:]) - self.x_ft_start[2:]

                    self.slices = []
                    self.weights = []

                    for i in range(2**(self.dim-3)):
                        j = 2*i
                        s = bin(j)[2:].zfill(self.dim-2)
                        flags = [int(c) for c in s]

                        slices = [slice(None, None, 1), slice(None, None, 1)]
                        sizes = [self.in_channels, self.out_channels]

                        for k, f in enumerate(flags):
                            start = 0 if f == 0 else self.modes_high_starts[k]
                            stop = self.modes_low_stops[k] if f == 0 else self.x_ft_shape[2+k]
                            size = stop-start

                            slices.append(slice(start, stop, 1))
                            sizes.append(size)

                        sizes = np.array(sizes)
                        if (sizes <= 0).any():
                            continue

                        self.slices.append(tuple(slices))
                        self.weights.append(nn.Parameter(self.scale * torch.rand(sizes.tolist(), dtype=torch.cfloat)))

                    # See the below comment in forward()
                    assert len(self.weights) > 0

        x_ft = self.drfftn(x)

        out_ft = torch.zeros_like(x_ft)

        # TODO: There is currently a bug caused here when not all workers have a slice of
        # the weights, due no tensors within the layer having requires_grad=True, and thus
        # the output x not getting propagated that value. Strange bug and a lot of ugly
        # structural changes needed to fix...
        for sl, w in zip(self.slices, self.weights):
            out_ft[sl] = self.compl_mulnd(x_ft[sl], w)

        x = self.dirfftn(out_ft).real

        return x