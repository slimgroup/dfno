from dfno import create_standard_partitions
from distdl.utilities.slicing import *
from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from mpi4py import MPI
from torch.utils.data import Dataset 

import azure.storage.blob
import distdl.nn as dnn
import h5py, zarr, os
import numpy as np 
import torch


class DistributedSleipnerDataset3D(Dataset):
    ''' Distributed Dataset class for flow data generated with OPM'''

    def __init__(self,
                 P_feat,
                 samples,
                 client,
                 container,
                 prefix,
                 shape,
                 normalize=True,
                 padding=None,
                 savepath=None,
                 filename=None,
                 keep_data=False):
        
        self.P_feat = P_feat
        self.samples = samples
        self.client = client
        self.container = container
        self.prefix = prefix
        self.normalize = normalize
        self.padding = padding
        self.shape = shape
        self.savepath = savepath
        self.keep_data = keep_data
        self.filename = filename
        if savepath is not None:
            self.cache = list()

            # Check if files were already downloaded
            files = os.listdir(savepath)
            for i in samples:
                filename_curr = f'{self.filename}_{i:04d}_{self.P_feat.rank:04d}.h5'
                if filename_curr in files:
                    self.cache.append(filename_curr)
        else:
            self.cache = None

        self.yStart = compute_start_index(P_feat.shape[2:], P_feat.index[2:], self.shape)[1]
        self.yEnd = compute_stop_index(P_feat.shape[2:], P_feat.index[2:], self.shape)[1]
        
        # Open the data file
        self.store = zarr.ABSStore(container=self.container, prefix=self.prefix, client=self.client)       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        # Read 
        i = int(self.samples[index])
        
        # If caching is used, check if data sample exists locally
        filename = f'{self.filename}_{i:04d}_{self.P_feat.rank:04d}.h5'
        if self.cache is not None and filename in self.cache:
            fid = h5py.File(os.path.join(self.savepath, filename), 'r')
            x = torch.tensor(np.array(fid['x']))
            y = torch.tensor(np.array(fid['y']))
            nt = y.shape[-1]
            fid.close()

        else:
            nx, ny, nz, nt = self.shape
            
            # Read data from Blob store
            data = {}
            #data['permxy'] = torch.tensor(np.array(zarr.core.Array(self.store, path='permxy_' + str(i))[:,self.yStart:self.yEnd,:]), dtype=torch.float32)       # XYZ
            data['permz'] = torch.tensor(np.array(zarr.core.Array(self.store, path='permz_' + str(i))[:,self.yStart:self.yEnd,:]), dtype=torch.float32)         # XYZ
            data['tops'] = torch.tensor(np.array(zarr.core.Array(self.store, path='tops_' + str(i))[:,self.yStart:self.yEnd]), dtype=torch.float32)            # XY
            data['sat'] = torch.tensor(np.array(zarr.core.Array(self.store, path='saturation_' + str(i))[:nt+1, :,self.yStart:self.yEnd,:]), dtype=torch.float32)     # TXYZ
            data['sat'] = data['sat'].permute(1,2,3,0)[:,:,:,1:]    # TXYZ -> XYZT
            nx, ny, nz, nt = data['sat'].shape  # overwrite w/ local shape
                    
            # Normalize between 0 and 1
            data['sat'][data['sat'] < 0] = 0    # clip negative values which shouldn't be there
            if self.normalize:
                for k in data:

                    # TODO: Assumes MPI backend
                    minval = data[k].min()
                    minval = self.P_feat._comm.allreduce(minval, op=MPI.MIN)
                    data[k] -= minval
                    maxval = data[k].max()
                    maxval = self.P_feat._comm.allreduce(maxval, op=MPI.MAX)
                    data[k] /= maxval

            # Reshape to C X Y Z T (differs from sequential implementation to avoid shuffle ops)
            #permxy = data['permxy'].view(1, nx, ny, nz, 1)
            permz = data['permz'].view(1, nx, ny, nz, 1)
            tops = data['tops'].view(1, nx, ny, 1, 1).repeat(1, 1, 1, nz, 1)
            y = data['sat'].view(1, nx, ny, nz, nt)

            x = torch.cat((
                #permxy,
                permz,
                tops
                ),
                axis=0
            )

            # Write file to disk for later use
            if self.cache is not None:
                fid = h5py.File(os.path.join(self.savepath, filename), 'w')
                fid.create_dataset('x', data=x)
                fid.create_dataset('y', data=y)
                fid.close()
                self.cache.append(filename)

        return x, y
