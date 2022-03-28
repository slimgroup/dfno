import distdl.nn as dnn
import torch
import torch.nn as nn

from distdl.utilities.torch import zero_volume_tensor
from dfno.utils import create_root_partition, create_standard_partitions
from gradient_test import gradient_test

class BroadcastedLinear(nn.Module):

    def __init__(self, P_x, in_features, out_features, dtype=torch.float32):
        super().__init__()

        self.P_x = P_x
        self.P_0 = create_root_partition(P_x)
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        
        if self.P_0.active:
            self.W = nn.Parameter(torch.rand(in_features, out_features, dtype=dtype))
            self.b = nn.Parameter(torch.rand(out_features, dtype=dtype))

        else:
            self.W = nn.Parameter(zero_volume_tensor(dtype=dtype))
            self.b = nn.Parameter(zero_volume_tensor(dtype=dtype))
        
        self.BW = dnn.Broadcast(self.P_0, self.P_x)
        self.Bb = dnn.Broadcast(self.P_0, self.P_x)

    def forward(self, x):
        W = self.BW(self.W)
        b = self.Bb(self.b)
        return W @ x + b

P_world, P_x, P_0 = create_standard_partitions((2,))
f = BroadcastedLinear(P_x, 16, 16, dtype=torch.float64)
input_shape = (16,)
all_ok = True
for r in gradient_test(f, input_shape):
    print(f'{str(r)}\n')
    if r.active:
        all_ok = all_ok and r.converged[0] and r.converged[1]
