import numpy as np
import torch
from mpi4py import MPI

# Tests if <F @ x1, y2> == <x1, F* @ y2>
# of equivalently <y1, y2> == <x1, x2>
# under the tight adjoint test
def check_adjoint_test_tight(P, x1, x2, y1, y2):

    x1 = x1.cpu()
    x2 = x2.cpu()
    y1 = y1.cpu()
    y2 = y2.cpu()

    local_results = np.zeros(6, dtype=np.complex64)
    global_results = np.zeros(6, dtype=np.complex64)

    # ||x1||^2
    local_results[0] = (torch.norm(x1)**2).numpy()
    # ||x2||^2 = ||F* @ y2||^2
    local_results[1] = (torch.norm(x2)**2).numpy()
    # <x1, x2> = <x1, F* @ y2>
    local_results[4] = np.array([torch.sum(torch.mul(x1, x2))])

    # ||y1||^2 = ||F @ x1 ||^2
    local_results[2] = (torch.norm(y1)**2).numpy()
    # ||y2||^2
    local_results[3] = (torch.norm(y2)**2).numpy()
    # <y1, y2> = <F @ x1, y2>
    local_results[5] = np.array([torch.sum(torch.mul(y1, y2))])

    # Reduce the norms and inner products
    P._comm.Reduce(local_results, global_results, op=MPI.SUM, root=0)

    # Because this is being computed in parallel, we risk that these norms
    # and inner products are not exactly equal, because the floating point
    # arithmetic is not commutative.  The only way to fix this is to gather
    # each component into rank 0 and have rank 0 do all of the arithmetic.
    # This will be tricky because we don't guarantee anything about the
    # shapes of the tensors.
    if(P.rank == 0):
        # Correct the norms from distributed calculation
        global_results[:4] = np.sqrt(global_results[:4])

        # Unpack the values
        norm_x1, norm_x2, norm_y1, norm_y2, ipx, ipy = global_results

        d = np.max([norm_y1*norm_y2, norm_x1*norm_x2])
        print(f"Adjoint test: {ipx/d} {ipy/d}")
        assert(np.isclose(ipx/d, ipy/d, atol=1e-3))
    else:
        # All other ranks pass the adjoint test
        assert(True)
