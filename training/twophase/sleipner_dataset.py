from dfno import create_standard_partitions
from distdl.utilities.slicing import assemble_slices
from distdl.utilities.tensor_decomposition import *
from distdl.utilities.torch import *
from dotenv import load_dotenv
from mpi4py import MPI
from torch.utils.data import Dataset 

import azure.storage.blob
import distdl.nn as dnn
import h5py, zarr, os
import numpy as np 
import torch

def compute_distribution_info(P, shape):
    info = {}
    ts = TensorStructure()
    ts.shape = shape
    info['shapes'] = compute_subtensor_shapes_balanced(ts, P.shape)
    info['starts'] = compute_subtensor_start_indices(info['shapes'])
    info['stops']  = compute_subtensor_stop_indices(info['shapes'])
    info['index']  = tuple(P.index)
    info['shape']  = info['shapes'][info['index']]
    info['start']  = info['starts'][info['index']]
    info['stop']   = info['stops'][info['index']]
    info['slice']  = assemble_slices(info['start'], info['stop'])
    return info

class DistributedSleipnerDataset3D(Dataset):
    ''' Distributed Dataset class for flow data generated with OPM'''

    def __init__(self,
                 P_x,
                 samples,
                 client,
                 container,
                 prefix,
                 shape,
                 nt,
                 normalize=True,
                 padding=None,
                 savepath=None,
                 filename=None,
                 keep_data=False,
                 P_y=None):
        
        self.P_x = P_x
        self.P_y = P_x if P_y is None else P_y
        self.samples = samples
        self.client = client
        self.container = container
        self.prefix = prefix
        self.nt = nt
        self.normalize = normalize
        self.padding = padding
        self.shape = shape
        self.savepath = savepath
        self.keep_data = keep_data
        self.filename = filename
        if savepath is not None:
            self.cache = list()
        else:
            self.cache = None

        self.x_info = compute_distribution_info(self.P_x, (1, 3, *shape, 1))
        self.y_info = compute_distribution_info(self.P_y, (1, 1, *shape, nt-1))
        self.y_pre_info = compute_distribution_info(self.P_y, (1, 1, *shape, nt))

        # Open the data file
        self.store = zarr.ABSStore(container=self.container, prefix=self.prefix, client=self.client)       

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        # Read 
        indices = self.samples[index]
        xs, ys = [], []
        
        for idx in indices:
            
            i = int(idx)

            # If caching is used, check if data sample exists locally
            filename = f'{self.filename}_{i:04d}_{self.P_x.rank:04d}.h5'
            if self.cache is not None and filename in self.cache:
                fid = h5py.File(os.path.join(self.savepath, filename), 'r')
                x = torch.tensor(fid['x'])
                sat = torch.tensor(fid['y'])
                fid.close()

            else:
                nx, ny, nz = self.x_info['shape'][2:5]

                sl_xyz  = tuple(self.x_info['slice'][2:5])
                sl_xy   = tuple(self.x_info['slice'][2:4])
                sl_txyz = list(self.y_pre_info['slice'][2:])
                sl_txyz = tuple([sl_txyz[-1]] + sl_txyz[0:-1])
                
                data = {}
                data['permxy'] = torch.tensor(np.array(zarr.core.Array(self.store, path='permxy_' + str(i))[sl_xyz]), dtype=torch.float32)       # XYZ
                data['permz'] = torch.tensor(np.array(zarr.core.Array(self.store, path='permz_' + str(i))[sl_xyz]), dtype=torch.float32)         # XYZ
                data['tops'] = torch.tensor(np.array(zarr.core.Array(self.store, path='tops_' + str(i))[sl_xy]), dtype=torch.float32)            # XY
                data['sat'] = torch.tensor(np.array(zarr.core.Array(self.store, path='saturation_' + str(i))[sl_txyz]), dtype=torch.float32)     # TXYZ
                data['pressure'] = torch.tensor(np.array(zarr.core.Array(self.store, path='pressure_' + str(i))[sl_txyz]), dtype=torch.float32)  # TXYZ
                
                # Permute
                data['sat'] = data['sat'].permute(1,2,3,0)[:,:,:,1:]              # TXYZ -> XYZT
                data['pressure'] = data['pressure'].permute(1,2,3,0)[:,:,:,1:]    # TXYZ -> XYZT

                # Normalize between 0 and 1
                if self.normalize:
                    for k in data:

                        # TODO: Assumes MPI backend
                        comm = self.P_y._comm if k in ['sat', 'pressure'] else self.P_x._comm
                        minval = data[k].min()
                        maxval = data[k].max()
                        minval = comm.allreduce(minval, op=MPI.MIN)
                        maxval = comm.allreduce(maxval, op=MPI.MAX)
                        data[k] -= minval; data[k] /= maxval

                # Reshape to C X Y Z (differs from sequential implementation to avoid shuffle ops)
                permxy = data['permxy'].view(1, nx, ny, nz)
                permz  = data['permz'].view(1, nx, ny, nz)
                tops   = data['tops'].view(1, nx, ny, 1).repeat(1, 1, 1, nz)

                # Reshape to X Y Z T
                sat      = data['sat'].view(nx, ny, nz, self.nt-1)
                pressure = data['pressure'].view(nx, ny, nz, self.nt-1)            

                # Padding
                if self.padding is not None:
                    raise ValueError('Padding distributed arrays is currently unsupported')
                    xpad, ypad, zpad = self.padding
                    permxy = torch.nn.functional.pad(permxy, (0,0,zpad,zpad,ypad,ypad,xpad,xpad))
                    permz = torch.nn.functional.pad(permz, (0,0,zpad,zpad,ypad,ypad,xpad,xpad))
                    tops = torch.nn.functional.pad(tops, (0,0,zpad,zpad,ypad,ypad,xpad,xpad))
                    sat = torch.nn.functional.pad(sat, (0,0,zpad,zpad,ypad,ypad,xpad,xpad))
                    pressure = torch.nn.functional.pad(pressure, (0,0,zpad,zpad,ypad,ypad,xpad,xpad))

                x = torch.cat((
                    permxy,
                    permz,
                    tops
                    ),
                    axis=0
                )
                y = torch.cat((
                    sat,
                    #pressure
                    ),
                    axis=0
                )

                x = x.unsqueeze(0).unsqueeze(-1)
                y = y.unsqueeze(0).unsqueeze(1)

                # Write file to disk for later use
                if self.cache is not None:
                    fid = h5py.File(os.path.join(self.savepath, filename), 'w')
                    fid.create_dataset('x', data=x)
                    fid.create_dataset('y', data=y)
                    fid.close()
                    self.cache.append(filename)

            xs.append(x)
            ys.append(y)

        return torch.cat(xs, axis=0), torch.cat(ys, axis=0)

    def close(self):
        if self.keep_data is False and self.cache is not None:
            print('Delete temp files.')
            for file in self.cache:
                os.system('rm ' + self.savepath + '/' + file)

#######################################################################################################################

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import plotly.graph_objects as go

    load_dotenv()

    P_x_shape = (1, 1, 2, 2, 1, 1)
    P_world, P_x, P_0 = create_standard_partitions(P_x_shape)

    # Computational grid
    nx = 60
    ny = 60
    nz = 64
    nt = 30

    container = os.environ['CONTAINER']
    data_path = os.environ['DATA_PATH']

    # Az storage client
    client = azure.storage.blob.ContainerClient(
        account_url=os.environ['ACCOUNT_URL'],
        container_name=container,
        credential=os.environ['SLEIPNER_CREDENTIALS']
        )

    # Data parameters
    in_channels = 3     # permeability xy, permeability z, topography
    out_channels = nt   # 24 time steps of saturation history

    # Data split
    num_train = 800
    num_valid = 100
    num_test = 100

    #  Train sample indices
    train_idx = torch.linspace(1, num_train, num_train, dtype=torch.int32).long()
    valid_idx = torch.linspace(num_train+1, num_train + num_valid, num_valid, dtype=torch.int32).long()
    test_idx = torch.linspace(num_train+num_valid+1, num_train+num_valid+num_test, num_test, dtype=torch.int32).long()

    # Sleipner dataset
    savepath = os.path.join(os.getcwd(), 'data')
    train_data = DistributedSleipnerDataset3D(P_x, train_idx, client, container, data_path, (nx, ny, nz), nt,
            normalize=True, savepath=savepath, filename=os.environ['TRAIN_FILENAME'])
    valid_data = DistributedSleipnerDataset3D(P_x, valid_idx, client, container, data_path, (nx, ny, nz), nt,
            normalize=True, savepath=savepath, filename=os.environ['TEST_FILENAME'])
    test_data = DistributedSleipnerDataset3D(P_x, test_idx, client, container, data_path, (nx, ny, nz), nt,
            normalize=True, savepath=savepath, filename=os.environ['VALID_FILENAME'])

    # Get sample data
    iterator = iter(train_data)
    x, y = next(iterator)

    print(f'index = {P_x.index}, x.shape = {x.shape}, y.shape = {y.shape}')

    G_x = dnn.DistributedTranspose(P_x, P_0)
    G_y = dnn.DistributedTranspose(P_x, P_0)

    x_global = G_x(x.view(1, *x.shape, 1))
    y_global = G_y(y.view(1, 1, *y.shape))

    if P_0.active:
        x_global = x_global.view((3, nx, ny, nz))
        y_global = y_global.view((nx, ny, nz, nt-1))

        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(np.transpose(x_global[0,:,30,:])); plt.title('Permeability')
        plt.subplot(1,3,2)
        plt.imshow(x_global[2,:,:,0]); plt.title('Topography')
        plt.subplot(1,3,3)
        plt.imshow(np.transpose(y_global[:,30,:,-1])); plt.title('Saturation i=30')
        plt.savefig(f'test.png')

        with open('y_global.npy', 'wb') as f:
            np.save(f, y_global.cpu().detach().numpy())

    # Delete temporary files
    train_data.close()
    valid_data.close()
    test_data.close()
