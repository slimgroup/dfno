import distdl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from collections import OrderedDict
from distdl.utilities.torch import *
from distdl.utilities.tensor_decomposition import *
from spectral_conv import DistributedSpectralConvNd
from torch import Tensor
from typing import List, Optional

Partition = distdl.backend.backend.Partition

class DistributedFNONd(nn.Module):

    def __init__(self,
                 P_x: Partition,
                 in_channels: int,
                 out_timesteps: int,
                 modes: List[int],
                 width: int,
                 num_blocks: int = 4) -> None:

        super(DistributedFNONd, self).__init__()

        self.P_x = P_x
        self.P_t_shape = np.copy(self.P_x.shape)
        self.P_t_shape[1] = self.P_x.shape[-1]
        self.P_t_shape[-1] = self.P_x.shape[1]
        self.P_t = self.P_x.create_cartesian_topology_partition(self.P_t_shape)
        self.P_y = P_x

        self.in_channels = in_channels
        self.out_timesteps = out_timesteps
        self.modes = modes
        self.width = width
        self.num_blocks = num_blocks

        if not self.P_x.active:
            return

        self.dim = P_x.dim

        self.si = ''.join(chr(i+97) for i in range(self.dim-2))
        self.channel_mul_einsum_equation = f'xy{self.si},yz->xz{self.si}'
        self.weight_mul_einsum_equation = f'xy{self.si},yz{self.si}->xz{self.si}'
        self.timestep_mul_einsum_equation = f'x{self.si}y,yz->x{self.si}z'

        self.convs = [DistributedSpectralConvNd(self.P_x, self.width, self.width, self.modes, P_y=self.P_x) for _ in range(self.num_blocks)]
        self.norms = [distdl.nn.DistributedBatchNorm(self.P_x, num_features=self.width) for _ in range(self.num_blocks)]
        self.weights = [nn.Parameter(torch.rand(self.width, self.width, *[1]*(self.dim-2))) for _ in range(self.num_blocks)]

        self.fc0 = nn.Parameter(torch.rand(self.in_channels, self.width))
        self.fc1 = nn.Parameter(torch.rand(self.width, 128))
        self.fc2 = nn.Parameter(torch.rand(128, 1))
        self.fc3 = None

        self.T0 = distdl.nn.DistributedTranspose(self.P_x, self.P_t)
        self.T1 = distdl.nn.DistributedTranspose(self.P_t, self.P_x)

    def forward(self, x: Tensor) -> Tensor:

        with torch.no_grad():
            if self.fc3 is None:
                self.fc3 = nn.Parameter(torch.rand(x.shape[-1], self.out_timesteps))

        if not self.P_x.active:
            return

        x = torch.einsum(self.channel_mul_einsum_equation, x, self.fc0)

        for C, N, w in zip(self.convs, self.norms, self.weights):
            x1 = C(x)
            x2 = torch.einsum(self.weight_mul_einsum_equation, x, w)
            x = x1 + x2
            x = N(x)
            x = F.gelu(x)

        x = torch.einsum(self.channel_mul_einsum_equation, x, self.fc1)
        x = F.gelu(x)
        x = torch.einsum(self.channel_mul_einsum_equation, x, self.fc2)
    
        x = self.T0(x)
        x = torch.einsum(self.timestep_mul_einsum_equation, x, self.fc3)
        x = F.gelu(x)
        x = self.T1(x)

        return x
