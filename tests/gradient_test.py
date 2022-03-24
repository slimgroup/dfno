from dataclasses import dataclass
from typing import Callable, Generator, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

@dataclass
class GradientTestResult:
    name: str
    active: bool
    converged: Tuple[bool, bool]
    convergence: Tuple[List[float], List[float]]
    steps: List[float]
    polyfit: Tuple[List[float], List[float]]

    def __str__(self):
        s = f'==== {self.name} ====\n'
        s += f'active: {self.active}\n'
        s += f'converged:\n'
        s += f'\tO(h):   {self.converged[0]}\n'
        s += f'\tO(h^2): {self.converged[1]}\n'
        s += f'convergence:\n'
        s += f'\tO(h)   err = {self.convergence[0]}\n'
        s += f'\tO(h^2) err = {self.convergence[1]}\n'
        s += f'steps:\n'
        s += f'\t h = {self.steps}\n'
        s += f'polyfit:\n'
        if self.active:
            s += f'\tO(h)   poly = {self.polyfit[0][0]}h + {self.polyfit[0][1]}\n'
            s += f'\tO(h^2) poly = {self.polyfit[1][0]}h + {self.polyfit[1][1]}'
        else:
            s += f'\tO(h)   poly = N/A\n'
            s += f'\tO(h^2) poly = N/A\n'
        return s

def gradient_test(f: nn.Module,
                  input_shape: List[int],
                  max_iter: int = 10,
                  dtype: torch.dtype = torch.float64) -> Generator[GradientTestResult, None, None]:

    def inner(p: nn.Parameter) -> GradientTestResult:

        def loss(x, y):
            try:
                p.grad.zero_()
            except AttributeError:
                pass
            return .5*torch.norm(f(x) - y)**2

        # Get the value that was stored in p initially
        p_init = p.data

        # Set intial weights and perturbation 
        p0 = 1 + torch.rand(*p.shape, dtype=p.dtype)
        dp = 1e-3*(1 + torch.rand(*p.shape, dtype=p.dtype))

        # Initial inputs
        x0 = 1 + torch.rand(*input_shape, dtype=dtype)
        x1 = 1 + torch.rand(*input_shape, dtype=dtype)

        # Intialize layer
        p.data = p0

        # Loss and gradient at x1
        with torch.no_grad():
            y0 = f(x0)

        f0 = loss(x1, y0)
        f0.backward()
        
        active = True
        try:
            g0 = p.grad.detach()
            gdx = torch.dot(dp.flatten(), g0.flatten())
            f0, gdx = f0.detach().item(), gdx.detach().item()
        except AttributeError:
            # In a distributed setting, it is not always the case that a parameter
            # has a gradient w.r.t the input
            active = False

        # Compute taylor error for varying step size
        err1 = []
        err2 = []
        hs = []
        h = 1

        for i in range(max_iter):

            # Perturb weight
            p.data = p0 + h*dp

            # Get f(x+dx))
            fk = loss(x1, y0)
            fk = float(fk.detach())
            
            # Only compute error terms if the parameter is active
            if active:
                # First order error
                # Per taylor, we have `f(x+dx) = f(x) + O(h)` so this should decrease linearly with h
                err1.append(abs(fk - f0))

                # Second order error
                # Per taylor, we have `f(x+dx) = f(x) + h*<g, dw> + O(h^2)` so this should decrease
                # quadratically with h
                err2.append(abs(fk - f0 - h*gdx))
            
            hs.append(h)
            h = h/2
        
        p1, p2 = [], []
        err1_converged, err2_converged = False, False
        if active:
            p1 = np.polyfit(np.log10(hs), np.log10(err1), 1)
            p2 = np.polyfit(np.log10(hs), np.log10(err2), 1)
            err1_converged = np.isclose(p1[0], 1.0, rtol=0.1)
            err2_converged = np.isclose(p2[0], 2.0, rtol=0.1)
            
        # Reset parameter once we are finished with it
        p.data = p_init

        return GradientTestResult('', active, (err1_converged, err2_converged), (err1, err2), hs, (p1, p2))
    
    for name, p in f.named_parameters():
        gt = inner(p)
        gt.name = name
        yield gt
