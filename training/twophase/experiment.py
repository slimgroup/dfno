from argparse import ArgumentParser
from dfno import create_standard_partitions, DistributedFNONd
from dfno.utils import generate_batch_indices, grid_like
from distdl.utilities.torch import *
from dotenv import load_dotenv
from sleipner_dataset import DistributedSleipnerDataset3D

import azure.storage.blob
import distdl.nn as dnn
import h5py
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn

load_dotenv()

# Distribution info
P_x_shape = (1, 1, 2, 2, 1, 1)
P_world, P_x, P_0 = create_standard_partitions(P_x_shape)
num_gpus = 4
device = torch.device(f'cuda:{P_x.rank}')

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

# Communicate unique save dir with each rank
B = dnn.Broadcast(P_0, P_x)
timestamp = torch.tensor([int(time.time())]) if P_0.active else zero_volume_tensor()
timestamp = B(timestamp).item()

# Computational grid
nx = 60
ny = 60
nz = 64
nt = 31 # There's currently a bug in the distributed FFT impl that requires this to be odd

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
num_train = 2
num_valid = 1
num_test = 1

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

# FNO parameters
width = 20
modes = (8, 8, 8, 8)
decomposition_order = 2
num_blocks = 4

# Training parameters
batch_size = 1
num_epochs = 1
shuffle = True
checkpoint_interval = 1
sample_batch_index = 0
out_dir = os.path.join(savepath, f'out_{timestamp}')

if P_0.active:
    os.makedirs(out_dir, exist_ok=False)
    print(f'created output directory: {out_dir}')
P_x._comm.Barrier()

network = DistributedFNONd(
    P_x,
    width,
    modes,
    nt-1,
    decomposition_order=decomposition_order,
    num_blocks=num_blocks,
    dtype=torch.float32,
    P_y=P_x
)

dummy = torch.rand(batch_size, *train_data.x_info['shape'][1:], dtype=torch.float32)
with torch.no_grad():
    _ = network(dummy)

network = network.to(device)

parameters = [p for p in network.parameters()]
criterion = dnn.DistributedMSELoss(P_x).to(device)
mse = dnn.DistributedMSELoss(P_x).to(device)
optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)

if P_0.active:
    steps = []
    train_accs = []
    test_accs = []

for i in range(num_epochs):

    network.train()
    batch_indices = generate_batch_indices(P_x, num_train, batch_size, shuffle=shuffle)
    train_loss = 0
    n_train_batch = 0

    for j, (a, b) in enumerate(batch_indices):
        
        t0 = time.time()
        x, y = train_data[a:b]
        x = x.to(device)
        y = y.to(device)

        y_hat = network(x)
        loss = criterion(y_hat, y)
        if P_0.active:
            print(f'epoch = {i}, batch = {j}, loss = {loss.item()}')
            train_loss += loss.item()
            n_train_batch += 1
        loss.backward()
        optimizer.step()

        P_x._comm.Barrier()
        t1 = time.time()
        print(f'epoch = {i}, batch = {j}, dt = {t1-t0}')

    if P_0.active:
        print(f'epoch = {i}, average train loss = {train_loss/n_train_batch}')
        steps.append(i)
        train_accs.append(train_loss/n_train_batch)
    
    P_x._comm.Barrier()

    network.eval()
    batch_indices = generate_batch_indices(P_x, num_test, batch_size, shuffle=False)
    test_loss = 0
    n_test_batch = 0
    y_true, y_pred = None, None

    for j, (a, b) in enumerate(batch_indices):
        with torch.no_grad():
            t0 = time.time()
            x, y = test_data[a:b]
            x = x.to(device)
            y = y.to(device)

            y_hat = network(x)
            loss = criterion(y_hat, y)
            if P_0.active:
                print(f'epoch = {i}, batch = {j}, test loss = {loss.item()}')
                test_loss += loss.item()
                n_test_batch += 1

            if j == sample_batch_index and (i+1) % checkpoint_interval == 0:
                y_true, y_pred = y.cpu().detach().numpy(), y_hat.cpu().detach().numpy()

    if P_0.active:
        print(f'epoch = {i}, average test loss = {test_loss/n_test_batch}')

    if (i+1) % checkpoint_interval == 0:
        filename = f'y_{i:04d}_{P_x.rank:04d}.h5'
        fid = h5py.File(os.path.join(out_dir, filename), 'w')
        fid.create_dataset('y_true', data=y_true)
        fid.create_dataset('y_pred', data=y_pred)
        fid.close()
        print(f'rank = {P_x.rank}, wrote predicition file: {filename}')

        model_path = os.path.join(out_dir, f'model_{j:04d}_{P_x.rank:04d}.pt')
        torch.save(network.state_dict(), model_path)
        print(f'rank = {P_x.rank}, saved model: {model_path}')

if P_0.active:
    print('training finished.')
