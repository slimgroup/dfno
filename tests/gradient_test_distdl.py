import distdl.nn as dnn
import gc
import torch
import torch.nn as nn

from dfno.utils import create_standard_partitions
from dfno import BroadcastedAffineOperator
from gradient_test import gradient_test

P_world, P_x, P_0 = create_standard_partitions((1, 2))
_, P_y, _ = create_standard_partitions((2, 1))

# This network passes gradcheck
f = nn.Sequential(
    nn.Linear(16, 16, dtype=torch.float64),
    dnn.DistributedTranspose(P_x, P_y),
    dnn.DistributedTranspose(P_y, P_x),
    nn.Linear(16, 16, dtype=torch.float64)
)

# local shape
input_shape = (2, 16)

# Run test
all_ok = True
for r in gradient_test(f, input_shape):
    if P_0.active:
        print(str(r))
    if r.active:
        all_ok = all_ok and r.converged[0] and r.converged[1]

P_x._comm.Barrier()

if all_ok:
    print(f'rank {P_x.rank} passed gradcheck 1')
else:
    print(f'rank {P_x.rank} failed gradcheck 1')

P_x._comm.Barrier()

# This network does not pass gradcheck... why?
# Could be an issue with the partition activity
# vs the grad activity or something... very weird.
f = nn.Sequential(
    nn.Linear(16, 16, dtype=torch.float64),
    dnn.DistributedTranspose(P_x, P_y),
    nn.Linear(32, 32, dtype=torch.float64),
    dnn.DistributedTranspose(P_y, P_x)
)

# local shape
input_shape = (2, 16)

# Run test
all_ok = True
for r in gradient_test(f, input_shape):
    if P_0.active:
        print(str(r))
    if r.active:
        all_ok = all_ok and r.converged[0] and r.converged[1]

P_x._comm.Barrier()

if all_ok:
    print(f'rank {P_x.rank} passed gradcheck 2')
else:
    print(f'rank {P_x.rank} failed gradcheck 2')
