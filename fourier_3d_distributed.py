import distdl
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from dfft import DXFFTN
from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from torch import Tensor

Partition = distdl.backend.backend.Partition

class DistributedSpectralConv3d(nn.Module):

    def __init__(self,
                 P_x: Partition,
                 in_channels: int,
                 out_channels: int,
                 modes1: int,
                 modes2: int,
                 modes3: int) -> None:

        super(DistributedSpectralConv3d, self).__init__()
    
        # Set passed parameters
        self.P_x = P_x

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = 1 / (in_channels * out_channels)
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        # Set up foward distributed real-valued fft (save the inverse transform setup until
        # input size is known)
        self.drfftn_transforms = [torch.fft.fftn, torch.fft.rfftn]
        self.drfftn_transform_kwargs = [{'dim': (-3, -2)}, {'dim': (-1,)}]
        self.drfftn = DXFFTN(self.P_x, dims=[-3, -2, -1], transforms=self.drfftn_transforms, transform_kwargs=self.drfftn_transform_kwargs, decomposition_order=2)

        self.dirfftn = None

        # Extract the last drfftn partition to compute weight shapes and slice indices
        self.P_fft = self.drfftn.partitions[-1]

        # Create the weight parameters on first pass through forward so the input
        # size is known
        self.weights1 = None
        self.weights2 = None
        self.weights3 = None
        self.weights4 = None

    def create_weight(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int) -> Tensor:
        '''Creates a fourier mode weight for cases with positive mode sizes'''

        if all(x > 0 for x in [modes1, modes2, modes3]):
            return nn.Parameter(torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        else:
            return zero_volume_tensor()
        
    def compl_mul3d(self, x: Tensor, weights: Tensor) -> Tensor:
        '''Complex matrix multiplication in the channel dimension'''

        return torch.einsum('bixyz,ioxyz->boxyz', x, weights)

    def forward(self, x: Tensor) -> Tensor:

        batchsize = x.shape[0]
        
        # Initialize weights and inverse fourier transform once we can compute
        # the shape of x in the fourier domain (i.e. on first pass through the
        # network)
        if self.dirfftn is None:
            
            # Get the global input shape
            self.x_local_structure = TensorStructure(x)
            self.x_global_structure = \
                    distdl.backend.backend.tensor_comm.assemble_global_tensor_structure(self.x_local_structure, self.P_x)
            
            # Construct distributed inverse real fft
            self.dirfftn_transforms = [torch.fft.ifftn, torch.fft.irfftn]
            self.dirfftn_transform_kwargs = [{'dim': (-3, -2)}, {'s': (self.x_global_structure.shape[-1],)}]
            self.dirfftn = DXFFTN(self.P_fft, P_y=self.P_x, dims=[-3, -2, -1], transforms=self.dirfftn_transforms, transform_kwargs=self.dirfftn_transform_kwargs, decomposition_order=2)

            # Setup weights

            # Get global structure of x after it has been lifted into the fourier domain
            self.x_fft_global_structure = TensorStructure()
            self.x_fft_global_structure.shape = np.copy(self.x_global_structure.shape)
            self.x_fft_global_structure.shape[-1] = self.x_fft_global_structure.shape[-1] // 2 + 1
            
            # Get the local indexing information for lifted x
            self.x_fft_shapes = compute_subtensor_shapes_balanced(self.x_fft_global_structure, self.P_fft.shape)
            self.x_fft_starts = compute_subtensor_start_indices(self.x_fft_shapes)
            self.x_fft_stops = compute_subtensor_stop_indices(self.x_fft_shapes)
            
            self.P_fft_index = tuple(self.P_fft.index)
            self.x_fft_shape = self.x_fft_shapes[self.P_fft_index]
            self.x_fft_start = self.x_fft_starts[self.P_fft_index]
            self.x_fft_stop = self.x_fft_stops[self.P_fft_index]
            
            # Compute the stop/start values of low/high modes on the distributed tensor,
            # in local index coordinates
            self.modes1_stop_low = min(self.modes1, self.x_fft_stop[-3]) - self.x_fft_start[-3]
            self.modes2_stop_low = min(self.modes2, self.x_fft_stop[-2]) - self.x_fft_start[-2]
            self.modes3_stop_low = min(self.modes3, self.x_fft_stop[-1]) - self.x_fft_start[-1]

            self.modes1_start_high = max(self.x_fft_global_structure.shape[-3] - self.modes1, self.x_fft_start[-3]) - self.x_fft_start[-3]
            self.modes2_start_high = max(self.x_fft_global_structure.shape[-2] - self.modes2, self.x_fft_start[-2]) - self.x_fft_start[-2]
            self.modes3_start_high = max(self.x_fft_global_structure.shape[-1] - self.modes3, self.x_fft_start[-1]) - self.x_fft_start[-1]
            
            # Compute the size of the mode slices on this rank. This may be zero or negative
            # if the mode slices do not overlap with the subtensor on this rank, in which
            # case no weight multiplication is performed in the forward/adjoint pass.
            self.modes1_size_low = self.modes1_stop_low
            self.modes2_size_low = self.modes2_stop_low
            self.modes3_size_low = self.modes3_stop_low

            self.modes1_size_high = self.x_fft_shape[-3] - self.modes1_start_high
            self.modes2_size_high = self.x_fft_shape[-2] - self.modes2_start_high
            self.modes3_size_high = self.x_fft_shape[-1] - self.modes3_start_high
            
            # Instantiate the weights, giving a zero volume tensor if no work needs to be done
            self.weights1 = self.create_weight(self.in_channels, self.out_channels, self.modes1_size_low,  self.modes2_size_low,  self.modes3_size_low)
            self.weights2 = self.create_weight(self.in_channels, self.out_channels, self.modes1_size_high, self.modes2_size_low,  self.modes3_size_low)
            self.weights3 = self.create_weight(self.in_channels, self.out_channels, self.modes1_size_low,  self.modes2_size_high, self.modes3_size_low)
            self.weights4 = self.create_weight(self.in_channels, self.out_channels, self.modes1_size_high, self.modes2_size_high, self.modes3_size_low)

            self.use_weights1 = np.prod(self.weights1.shape) > 0
            self.use_weights2 = np.prod(self.weights2.shape) > 0
            self.use_weights3 = np.prod(self.weights3.shape) > 0
            self.use_weights4 = np.prod(self.weights4.shape) > 0

        # Lift x to the fourier domain
        x_ft = self.drfftn(x)
        
        # Allocate output tensor
        out_ft = torch.zeros(batchsize, self.out_channels, self.x_fft_shape[-3], self.x_fft_shape[-2], self.x_fft_shape[-1], dtype=torch.cfloat, device=x.device)
        
        # Multiply by mode weightings if applicable on this rank
        if self.use_weights1:
            out_ft[:, :, :self.modes1_stop_low, :self.modes2_stop_low, :self.modes3_stop_low] = \
                    self.compl_mul3d(x_ft[:, :, :self.modes1_stop_low, :self.modes2_stop_low, :self.modes3_stop_low], self.weights1)

        if self.use_weights2:
            out_ft[:, :, self.modes1_start_high:, :self.modes2_stop_low, :self.modes3_stop_low] = \
                    self.compl_mul3d(x_ft[:, :, self.modes1_start_high:, :self.modes2_stop_low, :self.modes3_stop_low], self.weights2)

        if self.use_weights3:
            out_ft[:, :, :self.modes1_stop_low, self.modes2_start_high:, :self.modes3_stop_low] = \
                    self.compl_mul3d(x_ft[:, :, :self.modes1_stop_low, :self.modes2_stop_low, :self.modes3_stop_low], self.weights3)

        if self.use_weights4:
            out_ft[:, :, self.modes1_start_high:, self.modes2_start_high:, :self.modes3_stop_low] = \
                    self.compl_mul3d(x_ft[:, :, self.modes1_start_high:, self.modes2_start_high:, :self.modes3_stop_low], self.weights4)
        
        # Return to physical space
        x = self.dirfftn(x_ft).real

        return x

class DistributedFNO3d(nn.Module):

    def __init__(self, P_x: Partition, modes1: int, modes2: int, modes3: int, width: int) -> None:
        super(DistributedFNO3d, self).__init__()
        
        self.P_x = P_x
        
        # For now only support spatial distribution
        assert P_x.shape[0] == 1, 'Batch distribution is currently unsupported'
        assert P_x.shape[1] == 1, 'Channel distribution is currently unsupported'

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.padding = 6 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(13, self.width)
        # input channel is 12: the solution of the first 10 timesteps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t)
        
        # Set up layers
        self.conv0 = DistributedSpectralConv3d(self.P_x, self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = DistributedSpectralConv3d(self.P_x, self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv2 = DistributedSpectralConv3d(self.P_x, self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv3 = DistributedSpectralConv3d(self.P_x, self.width, self.width, self.modes1, self.modes2, self.modes3)

        self.w0 = distdl.nn.DistributedFeatureConv3d(self.P_x, self.width, self.width, 1)
        self.w1 = distdl.nn.DistributedFeatureConv3d(self.P_x, self.width, self.width, 1)
        self.w2 = distdl.nn.DistributedFeatureConv3d(self.P_x, self.width, self.width, 1)
        self.w3 = distdl.nn.DistributedFeatureConv3d(self.P_x, self.width, self.width, 1)

        self.bn0 = distdl.nn.DistributedBatchNorm(self.P_x, self.width)
        self.bn1 = distdl.nn.DistributedBatchNorm(self.P_x, self.width)
        self.bn2 = distdl.nn.DistributedBatchNorm(self.P_x, self.width)
        self.bn3 = distdl.nn.DistributedBatchNorm(self.P_x, self.width)
        
        self.fc1 = nn.Parameter(torch.randn(self.width, 128))
        self.fc2 = nn.Parameter(torch.randn(128, 1))

    def forward(self, x: Tensor) -> Tensor:

        # TODO: There is some strange padding operator here in the sequential
        # code. Need to implement that in a distributed fashion somehow. Also a
        # permutation of the input tensor, assuming due to the way data is stored?
        # A custom data loader is definitely needed.

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        # Do the linear layers using einsum() to minimize data movement. These
        # are independent of spatial dimensions, and batch and channel are
        # assumed to not be distributed.
        x = torch.einsum('bixyz,io->boxyz', x, self.fc1)
        x = F.gelu(x)
        x = torch.einsum('bixyz,io->boxyz', x, self.fc2)

        return x

if __name__ == '__main__':

    from mpi4py import MPI

    P_world = Partition(MPI.COMM_WORLD)
    P_0_base = P_world.create_partition_inclusive([0])
    P_0 = P_0_base.create_cartesian_topology_partition([1, 1, 1, 1, 1])
    P_x_base = P_world.create_partition_inclusive(np.arange(8))
    P_x = P_x_base.create_cartesian_topology_partition([1, 1, 2, 2, 2])
    
    batchsize = 20
    width = 20
    modes = 8
    sr = 32
    
    scatter = distdl.nn.DistributedTranspose(P_0, P_x)
    gather = distdl.nn.DistributedTranspose(P_x, P_0)
    sconv = DistributedSpectralConv3d(P_x, width, width, modes, modes, modes)

    if P_0.active:
        print('==== DistributedSpectralConv3d Info ====')

        print('== FFT Partitions ==')
        print('partition_dims -> partition')
        for ds, P in zip(sconv.drfftn.partition_dims, sconv.drfftn.partitions):
            print(f'{ds} -> {P.shape}')
        
    P_x._comm.Barrier()

    if P_0.active:
        
        axes = [np.arange(batchsize), np.arange(width), *[np.linspace(0, 1, sr) for _ in range(3)]]
        Xs = np.meshgrid(*axes)

        Z = np.zeros_like(Xs[0], dtype=np.float32)
        for i, X in enumerate(Xs[-3:]):
            Z += np.sin(2*np.pi*(i+1)*X)

        Z = Tensor(Z)
    
    else:
        Z = zero_volume_tensor()

    Z = scatter(Z)
    Z = sconv(Z)
    
    if P_0.active:
        print('==== Output Shape ====')
        print('rank -> output shape')
    
    P_x._comm.Barrier()

    print(f'{P_x.rank} -> {Z.shape}')
