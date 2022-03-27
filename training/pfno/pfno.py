import numpy as np
import torch, distdl, math
from mpi4py import MPI
from distdl.backends.mpi.partition import MPIPartition
from distdl.nn.repartition import Repartition
from distdl.nn.broadcast import Broadcast
from distdl.utilities.torch import zero_volume_tensor
from distdl.nn import DistributedTranspose
from distdl.utilities.slicing import *
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Linear(distdl.nn.Module):

    def __init__(self, P_root, P_feat, channel_in, channel_out):
        super(Linear, self).__init__()
        device = torch.device(f'cuda:{P_feat.rank}')

        if P_root.active:
            w_scale = 1.0 / channel_in / channel_out
            b_scale = 1.0 / channel_out
            self.w = torch.nn.Parameter(w_scale * torch.rand(channel_in, channel_out, 1, 1, 1, 1, device=device))
            self.b = torch.nn.Parameter(b_scale * torch.rand(1, channel_out, 1, 1, 1, 1, device=device))
        else:
            self.register_buffer('w', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('b', torch.nn.Parameter(zero_volume_tensor(device=device)))

        self.broadcast_weights = Broadcast(P_root, P_feat)
        self.broadcast_bias = Broadcast(P_root, P_feat)

    def forward(self, x):
        # Assumes that x is already partitioned across P_feat
        # Weights are stored on the root partition only

        # Broadcast weights to all workers
        w = self.broadcast_weights(self.w)
        b = self.broadcast_bias(self.b)

        # Linear encoding
        x = torch.einsum("bixyzt,ioxyzt->boxyzt", x, w) + b
        return x


class RFFT4D(distdl.nn.Module):

    def __init__(self, P_in, P_out):
        super(RFFT4D, self).__init__()

        self.transpose = Repartition(P_in, P_out)
        self.fft = torch.fft.fftn
        self.rfft = torch.fft.rfftn

    def forward(self, x):
        x = self.rfft(x, dim=(3,4,5))
        x = self.transpose(x)
        x = self.fft(x, dim=(2))
        return x


class IRFFT4D(distdl.nn.Module):

    def __init__(self, P_in, P_out):
        super(IRFFT4D, self).__init__()

        self.transpose = Repartition(P_in, P_out)
        self.fft = torch.fft.ifftn
        self.rfft = torch.fft.irfftn

    def forward(self, x):
        x = self.fft(x, dim=(2))
        x = self.transpose(x)
        x = self.rfft(x, dim=(3,4,5))
        return x


class SpectralConv(distdl.nn.Module):

    def __init__(self, P_feat, channel_hidden, global_shape, num_k):
        super(SpectralConv, self).__init__()
        device = torch.device(f'cuda:{P_feat.rank}')
        
        # Compute global indices
        xstart_index = compute_start_index(P_feat.shape[2:], P_feat.index[2:], global_shape)[0]
        xstop_index = compute_stop_index(P_feat.shape[2:], P_feat.index[2:], global_shape)[0]

        # Intersection between local slice and x mode low
        self.xlow_start_local, self.xlow_end_local, self.xlow_shape_local = compute_intersection(
            xstart_index, xstop_index, 0, num_k[0])

        # Intersection between local slice and x mode high
        self.xhigh_start_local, self.xhigh_end_local, self.xhigh_shape_local = compute_intersection(
            xstart_index, xstop_index, global_shape[0]-num_k[0], global_shape[0])

        # Remaining modes
        self.num_ky = num_k[1]
        self.num_kz = num_k[2]
        self.num_kw = num_k[3]

        # Initialize modes
        scaler = 1.0 / channel_hidden**2
        if self.xlow_shape_local > 0:
            self.w1 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xlow_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
            self.w2 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xlow_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
            self.w3 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xlow_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
            self.w4 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xlow_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
        else:
            self.register_buffer('w1', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('w2', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('w3', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('w4', torch.nn.Parameter(zero_volume_tensor(device=device)))

        if self.xhigh_shape_local > 0:
            
            self.w5 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xhigh_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
            self.w6 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xhigh_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
            self.w7 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xhigh_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
            self.w8 = torch.nn.Parameter(scaler * torch.rand(channel_hidden, channel_hidden, self.xhigh_shape_local, *num_k[1:], 
                device=device, dtype=torch.complex64))
        else:
            self.register_buffer('w5', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('w6', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('w7', torch.nn.Parameter(zero_volume_tensor(device=device)))
            self.register_buffer('w8', torch.nn.Parameter(zero_volume_tensor(device=device)))

    def forward(self, x):

        # Output tensor
        y = torch.clone(x)*0
        
        if self.xlow_shape_local > 0:
            y[:, :, :self.xlow_shape_local, :self.num_ky, :self.num_kz, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt", 
                x[:, :, :self.xlow_shape_local, :self.num_ky, :self.num_kz, :self.num_kw], self.w1)
            y[:, :, :self.xlow_shape_local, -self.num_ky:, :self.num_kz, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, :self.xlow_shape_local, -self.num_ky:, :self.num_kz, :self.num_kw], self.w2)
            y[:, :, :self.xlow_shape_local, :self.num_ky, -self.num_kz:, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, :self.xlow_shape_local, :self.num_ky, -self.num_kz:, :self.num_kw], self.w3)
            y[:, :, :self.xlow_shape_local, -self.num_ky:, -self.num_kz:, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, :self.xlow_shape_local, -self.num_ky:, -self.num_kz:, :self.num_kw], self.w4)

        if self.xhigh_shape_local > 0:
            y[:, :, -self.xhigh_shape_local:, :self.num_ky, :self.num_kz, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, -self.xhigh_shape_local:, :self.num_ky, :self.num_kz, :self.num_kw], self.w5)
            y[:, :, -self.xhigh_shape_local:, -self.num_ky:, :self.num_kz, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, -self.xhigh_shape_local:, -self.num_ky:, :self.num_kz, :self.num_kw], self.w6)
            y[:, :, -self.xhigh_shape_local:, :self.num_ky, -self.num_kz:, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, -self.xhigh_shape_local:, :self.num_ky, -self.num_kz:, :self.num_kw], self.w7)
            y[:, :, -self.xhigh_shape_local:, -self.num_ky:, -self.num_kz:, :self.num_kw] = torch.einsum("bixyzt,ioxyzt->boxyzt",
                x[:, :, -self.xhigh_shape_local:, -self.num_ky:, -self.num_kz:, :self.num_kw], self.w8)

        return y


class FNOLayer4D(distdl.nn.Module):

    def __init__(self, P_root, P_x, P_y, channel_hidden, shape, num_k):
        super(FNOLayer4D, self).__init__()

        self.fft = RFFT4D(P_x, P_y)
        self.spectral_conv = SpectralConv(P_y, channel_hidden, shape, num_k)
        self.ifft = IRFFT4D(P_y, P_x)
        self.linear = Linear(P_root, P_x, channel_hidden, channel_hidden)

    def forward(self, x):

        xb = self.linear(x)
        x = self.fft(x)
        x = self.spectral_conv(x)
        x = self.ifft(x)
        x = F.relu(x + xb)

        return x


class ParallelFNO4d(distdl.nn.Module):

    def __init__(self, P_world, P_root, P_x, P_y, channel_in, channel_hidden, channel_out, shape, num_k, init_weights=True):
        super(ParallelFNO4d, self).__init__()
        P_world._comm.Barrier()

        # Encoder
        self.encoder1 = Linear(P_root, P_x, channel_in, channel_hidden // 2)
        self.encoder2 = Linear(P_root, P_x, channel_hidden // 2, channel_hidden)
        self.fno1 = FNOLayer4D(P_root, P_x, P_y, channel_hidden, shape, num_k)
        self.fno2 = FNOLayer4D(P_root, P_x, P_y, channel_hidden, shape, num_k)
        self.fno3 = FNOLayer4D(P_root, P_x, P_y, channel_hidden, shape, num_k)
        self.fno4 = FNOLayer4D(P_root, P_x, P_y, channel_hidden, shape, num_k)
        self.decoder1 = Linear(P_root, P_x, channel_hidden, channel_hidden // 2)
        self.decoder2 = Linear(P_root, P_x, channel_hidden // 2, channel_out)

    def forward(self, x):
        x = F.relu(self.encoder1(x))
        x = F.relu(self.encoder2(x))
        x = self.fno1(x)
        x = self.fno2(x)
        x = self.fno3(x)
        x = self.fno4(x)
        x = F.relu(self.decoder1(x))
        x = self.decoder2(x)
        return x



#######################################################################################################################

if __name__ == '__main__':

    # Init MPI
    P_world = MPIPartition(MPI.COMM_WORLD)
    P_world._comm.Barrier()
    n = P_world.shape[0]

    # Master worker partition with 6 dimensions ( N C X Y Z T )
    root_shape = (1, 1, 1, 1, 1, 1)
    P_root_base = P_world.create_partition_inclusive([0])
    P_root = P_root_base.create_cartesian_topology_partition(root_shape)

    # Distributed paritions
    feat_workers = np.arange(0, n)
    P_feat_base = P_world.create_partition_inclusive(feat_workers)
    P_x = P_feat_base.create_cartesian_topology_partition((1,1,n,1,1,1))
    #P_y = P_feat_base.create_cartesian_topology_partition((1,1,n,1,1,1))

    # Cuda
    device = torch.device(f'cuda:{P_x.rank}')

    # Reproducibility
    torch.manual_seed(P_x.rank + 123)
    np.random.seed(P_x.rank + 123)

    # Data dimensions
    nb = 1
    shape = (60, 60, 64, 30)    # X Y Z T
    num_train = 1
    num_valid = 1

    # Network dimensions
    channel_in = 3
    channel_hidden = 20
    channel_out = 1
    num_k = (2, 2, 2, 8)

    # Scatter data
    scatter = Repartition(P_root, P_x)
    gather = Repartition(P_x, P_root)

    # Data
    x = zero_volume_tensor()
    if P_root.active:
        x = torch.randn(nb, channel_hidden, shape[0], shape[1], shape[2], shape[3] // 2 + 1, dtype=torch.complex64)
    x = scatter(x).to(device)

    specconv = SpectralConv(P_x, channel_hidden, shape, num_k).to(device)

    y = specconv(x)
    x_ = gather(y).detach().cpu()

    if P_root.active:
        x_ = torch.real(x_)
        print(x_.shape)
        plt.imshow(x_[0,0,0,:,:,0]); plt.savefig('x')
        plt.imshow(x_[0,0,:,0,:,0]); plt.savefig('y')
        plt.imshow(x_[0,0,:,:,0,0]); plt.savefig('z')
        plt.imshow(x_[0,0,:,0,0,:]); plt.savefig('t')