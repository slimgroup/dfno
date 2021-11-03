import distdl
import distdl.nn as dnn
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import torch
import torch.nn as nn

from argparse import ArgumentParser
from distdl.utilities.torch import *
from dfno import *
from mat73 import loadmat
from matplotlib.animation import FuncAnimation
from mpi4py import MPI
from pathlib import Path
from scipy import io

Partition = distdl.backend.backend.Partition

parser = ArgumentParser()
parser.add_argument('--input',                  '-i',  type=Path)
parser.add_argument('--partition-shape',        '-ps', type=int,   default=(1,1,2,2,1), nargs=5)
parser.add_argument('--num-data',               '-nd', type=int,   default=1000)
parser.add_argument('--sampling-rate',          '-sr', type=int,   default=1)
parser.add_argument('--in-timesteps',           '-it', type=int,   default=10)
parser.add_argument('--out-timesteps',          '-ot', type=int,   default=40)
parser.add_argument('--device',                 '-d',  type=str,   default='cpu')
parser.add_argument('--num-gpus',               '-ng', type=int,   default=1)
parser.add_argument('--train-split',            '-ts', type=float, default=0.8)
parser.add_argument('--width',                  '-w',  type=int,   default=20)
parser.add_argument('--modes',                  '-m',  type=int,   default=(4, 4, 4), nargs=3)
parser.add_argument('--decomposition-order',    '-do', type=int,   default=1)
parser.add_argument('--num-blocks',             '-nb', type=int,   default=4)
parser.add_argument('--num-epochs',             '-ne', type=int,   default=500)
parser.add_argument('--batch-size',             '-bs', type=int,   default=10)
parser.add_argument('--checkpoint-interval',    '-ci', type=int,   default=25)
parser.add_argument('--generate-visualization', '-gv', action='store_true')

args = parser.parse_args()

if np.prod(args.partition_shape) != MPI.COMM_WORLD.size:
    raise ValueError(f'The number of processes {MPI.COMM_WORLD.size} does not match the partition shape {args.partition_shape}.')


P_world, P_x, P_0 = create_standard_partitions(args.partition_shape)
dim = P_x.dim
if args.device == 'cuda':
    device = torch.device(f'cuda:{P_x.rank % args.num_gpus}')
else:
    device = torch.device(args.device)

torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

B = dnn.Broadcast(P_0, P_x)
timestamp = torch.tensor([int(time.time())]) if P_0.active else zero_volume_tensor()
timestamp = B(timestamp).item()

torch.set_anomaly_enabled(True)

out_dir = Path(f'data/{args.input.stem}_{timestamp}')
if P_0.active:
    os.makedirs(out_dir)
    print(f'created output directory: {out_dir.resolve()}')

if P_0.active:
    #u = torch.rand(args.num_data, 1, 64, 64, args.in_timesteps+args.out_timesteps, device=device, dtype=torch.float32)
    u = torch.tensor(loadmat(args.input)['u'], dtype=torch.float32)[:args.num_data].unsqueeze(1).to(device)
    x_slice = (slice(None, args.num_data, 1), slice(None, None, 1), *[slice(None, None, args.sampling_rate)]*(dim-3), slice(None, args.in_timesteps, 1))
    y_slice = (slice(None, args.num_data, 1), slice(None, None, 1), *[slice(None, None, args.sampling_rate)]*(dim-3), slice(args.in_timesteps, args.in_timesteps+args.out_timesteps, 1))

    data = {}
    x, data['mu_x'], data['std_x'] = unit_guassian_normalize(u[x_slice])
    y, data['mu_y'], data['std_y'] = unit_guassian_normalize(u[y_slice])

    split_index = int(args.train_split*args.num_data)
    data['x_train'] = x[:split_index, ...]
    data['x_test']  = x[split_index:, ...]
    data['y_train'] = y[:split_index, ...]
    data['y_test']  = y[split_index:, ...]

    for k, v in data.items():
        print(f'{k}.shape = {v.shape}')

else:
    data = {}
    data['x_train'] = zero_volume_tensor(device=device)
    data['x_test'] = zero_volume_tensor(device=device)
    data['y_train'] = zero_volume_tensor(device=device)
    data['y_test'] = zero_volume_tensor(device=device)
    data['mu_x'] = zero_volume_tensor(device=device)
    data['std_x'] = zero_volume_tensor(device=device)
    data['mu_y'] = zero_volume_tensor(device=device)
    data['std_y'] = zero_volume_tensor(device=device)

for k, v in sorted(data.items(), key=lambda i: i[0]):
    S = dnn.DistributedTranspose(P_0, P_x)
    vars()[k] = S(v)
del data

print(f'index = {P_x.index}, x_train.shape = {x_train.shape}')
print(f'index = {P_x.index}, x_test.shape  = {x_test.shape}')
print(f'index = {P_x.index}, mu_x.shape    = {mu_x.shape}')
print(f'index = {P_x.index}, std_x.shape   = {std_x.shape}')
print(f'index = {P_x.index}, y_train.shape = {y_train.shape}')
print(f'index = {P_x.index}, y_test.shape  = {y_test.shape}')
print(f'index = {P_x.index}, mu_y.shape    = {mu_y.shape}')
print(f'index = {P_x.index}, std_y.shape   = {std_y.shape}')

x_train.requires_grad = True
y_train.requires_grad = True

network = DistributedFNONd(P_x,
                           args.width,
                           args.modes,
                           args.out_timesteps,
                           decomposition_order=args.decomposition_order,
                           num_blocks=args.num_blocks,
                           #device=device,
                           dtype=x_train.dtype,
                           P_y=P_x)

dummy = torch.rand(args.batch_size, *x_train.shape[1:], dtype=x_train.dtype)
with torch.no_grad():
    _ = network(dummy)

network = network.to(device)

parameters = [p for p in network.parameters()]
criterion = dnn.DistributedMSELoss(P_x).to(device)
mse = dnn.DistributedMSELoss(P_x).to(device)
optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)


if P_0.active and args.generate_visualization:
    steps = []
    train_accs = []
    test_accs = []

for i in range(args.num_epochs):
    network.train()
    batch_indices = generate_batch_indices(P_x, x_train.shape[0], args.batch_size, shuffle=True)
    train_loss = 0.0
    n_train_batch = 0.0
    for j, (a, b) in enumerate(batch_indices):
        optimizer.zero_grad()
        x = x_train[a:b]
        y = y_train[a:b]
        y_hat = network(x)
        y = unit_gaussian_denormalize(y, mu_y, std_y)
        y_hat = unit_gaussian_denormalize(y_hat, mu_y, std_y)
        loss = criterion(y_hat, y)
        if P_0.active:
            print(f'epoch = {i}, batch = {j}, loss = {loss.item()}')
            train_loss += loss.item()
            n_train_batch += 1
        loss.backward()
        optimizer.step()

    if P_0.active:
        print(f'epoch = {i}, average train loss = {train_loss/n_train_batch}')
        steps.append(i)
        train_accs.append(train_loss/n_train_batch)

    network.eval()
    with torch.no_grad():
        test_loss, test_mse = 0.0, 0.0
        y_true, y_pred = [], []
        batch_indices = generate_batch_indices(P_x, x_test.shape[0], args.batch_size, shuffle=False)
        n_test_batch = 0
        for j, (a, b) in enumerate(batch_indices):
            x = x_test[a:b]
            y = y_test[a:b]
            y_hat = network(x)
            y = unit_gaussian_denormalize(y, mu_y, std_y)
            y_hat = unit_gaussian_denormalize(y_hat, mu_y, std_y)
            loss = criterion(y_hat, y)
            mse_loss = mse(y_hat, y)
            test_loss += loss.item()
            test_mse += mse_loss.item()
            y_true.append(y)
            y_pred.append(y_hat)
            n_test_batch += 1
    
    if P_0.active:
        print(f'average test loss = {test_loss/n_test_batch}')
        print(f'average test mse  = {test_mse/n_test_batch}')
        test_accs.append(test_loss/n_test_batch)

    j = i+1
    if j % args.checkpoint_interval == 0:
        with torch.no_grad():
            model_path = out_dir.joinpath(f'model_{j:04d}_{P_x.rank:04d}.pt')
            torch.save(network.state_dict(), model_path)
            print(f'saved model: {model_path.resolve()}')

            y_true = torch.cat(tuple(y_true))
            y_pred = torch.cat(tuple(y_pred))
            mdict = {'y_true': y_true, 'y_pred': y_pred}
            mat_path = out_dir.joinpath(f'mat_{j:04d}_{P_x.rank:04d}.mat')
            io.savemat(mat_path, mdict)
            print(f'saved mat: {mat_path.resolve()}')
            
            if args.generate_visualization:
                G = dnn.DistributedTranspose(P_x, P_0)
                y_true = G(y_true).cpu().detach().numpy()
                y_pred = G(y_pred).cpu().detach().numpy()

                if P_0.active:
                    fig = plt.figure()
                    ax1 = fig.add_subplot(121)
                    ax2 = fig.add_subplot(122)
                    im1 = ax1.imshow(np.squeeze(y_true[0, :, :, :, 0]), animated=True)
                    im2 = ax2.imshow(np.squeeze(y_pred[0, :, :, :, 0]), animated=True)

                    def animate(k):
                        im1.set_data(np.squeeze(y_true[0, :, :, :, k]))
                        im2.set_data(np.squeeze(y_pred[0, :, :, :, k]))
                        return (im1, im2)

                    anim_path = out_dir.joinpath(f'anim_{j:04d}.gif')
                    ax1.title.set_text(r'$y_{true}$')
                    ax2.title.set_text(r'$y_{pred}$')
                    plt.axis('on')
                    anim = FuncAnimation(fig, animate, frames=args.out_timesteps, repeat=True)
                    anim.save(anim_path)
                    print(f'saved animation: {anim_path.resolve()}')

                    curve_path = out_dir.joinpath(f'curves_{j:04d}.png')
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    ax.plot(steps, train_accs, label='Average Train Loss')
                    ax.plot(steps, test_accs, label='Average Test Loss')
                    plt.axis('on')
                    plt.legend()
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.savefig(curve_path)
                    print(f'saved training curve plot: {curve_path.resolve()}')
