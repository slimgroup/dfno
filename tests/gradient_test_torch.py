import torch.nn as nn

from gradient_test import gradient_test

f = nn.Linear(10, 10)
gradient_test(f, (10, 10), max_iter=10)
