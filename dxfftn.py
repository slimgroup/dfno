import distdl
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from torch import Tensor
from typing import Optional

Partition = distdl.backend.backend.Partition

class DXFFTN(nn.Module):

    def __init__(self,
                 P_x: Partition,
                 info: OrderedDict,
                 P_y: Optional[Partition] = None) -> None:

        super(DXFFTN, self).__init__()

        self.P_x = P_x
        self.info = info

        if not self.P_x.active:
            self.P_y = P_x
            return

        self.transforms = []
        self.transform_kwargs = []
        self.partitions = [self.P_x]
        self.transposes = []

        for k, v in info.items():

            transform = v['transform']
            transform_kwargs = v['transform_kwargs']
            repartition_dims = v['repartition_dims']

            self.transforms.append(transform)
            self.transform_kwargs.append(transform_kwargs)

            shape = np.copy(self.P_x.shape)

            for i, d in enumerate(k):
                rpd = repartition_dims[i%len(repartition_dims)]
                shape[rpd] *= self.P_x.shape[d]
                shape[d] = 1

            self.partitions.append(self.P_x.create_cartesian_topology_partition(shape))
            self.transposes.append(distdl.nn.DistributedTranspose(self.partitions[-2], self.partitions[-1]))

        self.P_y = self.partitions[-1] if P_y is None else P_y
        self.transpose_out = distdl.nn.DistributedTranspose(self.partitions[-1], self.P_y)

    def forward(self, x: Tensor) -> Tensor:

        if not self.P_x.active:
            return x

        for f, kwargs, T in zip(self.transforms, self.transform_kwargs, self.transposes):
            x = T(x)
            x = f(x, **kwargs)

        x = self.transpose_out(x)
        return x