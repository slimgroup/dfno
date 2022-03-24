from dataclasses import dataclass
from torch import Tensor
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

@dataclass
class GradientTestResult:
    converged: bool
    convergence: Tuple[List[float], List[float], List[float]]

    def __str__(self):
        return f'{self.converged}, {self.convergence[0]}, {self.convergence[1]}, {self.convergence[2]}'

def gradient_test(f: Callable[[Tensor], Tensor], input_shape: List[int], max_iter: int = 10) -> GradientTestResult:
    
    n = len(input_shape)
    L = nn.MSELoss()

    x  = torch.rand(*input_shape)
    x0 = x+ 1**(-n)*torch.rand(*input_shape)
    dx = x-x0

    x.requires_grad = True
    x0.requires_grad = True

    y  = f(x)
    y0 = f(x0)
    dy = y-y0
    y.backward(dy)

    f0 = L(y, y0).item()

    x_grad = x.grad.detach()
    x = x.detach()
    x0 = x0.detach()
    dx = dx.detach()

    err1, err2, hs = [], [], []
    h = 1e-2

    def ip(p, q):
        return torch.inner(p.flatten(), q.flatten()).cpu().item()

    for i in range(max_iter):
        x_new = x0+h*dx
        y_new = f(x_new)
        fi = L(y, y_new).item()
        
        err1.append(abs(fi-f0))
        err2.append(abs(fi-f0 - h*ip(dx, x_grad)))
        hs.append(h)

        print(h, fi, f0, h, ip(x_grad, dx), fi-f0, h*ip(x_grad, dx), fi-f0 - h*ip(x_grad, dx))

        h /= 2

    p1 = np.polyfit(np.log10(hs), np.log10(err1), 1)
    p2 = np.polyfit(np.log10(hs), np.log10(err2), 1)
    print(p1, p2)
