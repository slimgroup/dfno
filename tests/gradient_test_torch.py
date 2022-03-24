from gradient_test import gradient_test

import gc
import torch
import torch.nn as nn

input_shape = (16, 16)
f = nn.Sequential(
        nn.Linear(16, 16, dtype=torch.float64),
        nn.Linear(16, 16, dtype=torch.float64)
)

# Initialize lazy parameters
with torch.no_grad():
    x = torch.rand(*input_shape, dtype=torch.float64, device=torch.device('cpu'))
    y = f(x)
    del x
    del y
    gc.collect()

# Run test
all_ok = True
for r in gradient_test(f, input_shape):
    print(str(r))
    all_ok = all_ok and r.converged[0] and r.converged[1]

if all_ok:
    print(f'passed gradcheck')
else:
    print(f'failed gradcheck')
