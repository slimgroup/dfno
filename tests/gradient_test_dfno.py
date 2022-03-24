from dfno.utils import create_standard_partitions
from dfno import DistributedFNONd
from gradient_test import gradient_test

import gc
import torch

# Run test on 16x16 spatial input distributed over 1x1x2x2x1 partition
P_world, P_x, P_0 = create_standard_partitions((1, 1, 2, 2, 1))
input_shape = (1, 1, 16, 16, 1)
f = DistributedFNONd(P_x=P_x,
                     width=20,
                     modes=(4, 4, 4),
                     out_timesteps=16,
                     decomposition_order=1,
                     num_blocks=4,
                     device=torch.device('cpu'),
                     dtype=torch.float64,
                     P_y=P_x)

# Initialize lazy parameters
x = torch.rand(*input_shape, dtype=torch.float64, device=torch.device('cpu'))
y = f(x)
del x
del y
gc.collect()

# Run test
all_ok = True
for r in gradient_test(f, input_shape):
    if P_0.active:
        print(str(r))
    if r.active:
        all_ok = all_ok and r.converged[0] and r.converged[1]

if all_ok:
    print(f'rank {P_x.rank} passed gradcheck')
else:
    print(f'rank {P_x.rank} passed gradcheck')
