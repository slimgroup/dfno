import distdl, torch, os, time, h5py
import numpy as np
import azure.storage.blob
import matplotlib.pyplot as plt
from mpi4py import MPI
from pfno import ParallelFNO, create_standard_partitions
from distdl.backends.mpi.partition import MPIPartition
from sleipner_dataset import DistributedSleipnerDataset3D
from distdl.nn.repartition import Repartition
from dotenv import load_dotenv
from loss import DistributedRelativeLpLoss

# this fails when placed at the top?? Torch must be setting cuda version or something...
import cupy

# Load env vars
load_dotenv()

# Partitions
P_world, P_x, P_root = create_standard_partitions((1, 1, 2, 1, 1, 1))
num_gpus = 2
device_ordinal = P_x.rank % num_gpus
device = torch.device(f'cuda:{device_ordinal}')
dtype = torch.float32

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

# Data dimensions
nb = 1
shape = (60, 60, 64, 10)    # X Y Z T
num_train = 1
num_valid = 1

# Network dimensions
channel_in = 3
width = 20
channel_out = 1
modes = (12, 12, 12, 5)

# Data store
container = os.environ['CONTAINER']
data_path = os.environ['DATA_PATH']

client = azure.storage.blob.ContainerClient(
    account_url=os.environ['ACCOUNT_URL'],
    container_name=container,
    credential=os.environ['SLEIPNER_CREDENTIALS']
    )

# Training dataset
train_idx = torch.linspace(1, num_train, num_train, dtype=torch.int32).long()
train_data = DistributedSleipnerDataset3D(P_x, train_idx, client, container, data_path, shape,
        normalize=True)

# Validation dataset
val_idx = torch.linspace(num_train, num_train+num_valid, num_valid, dtype=torch.int32).long()
valid_data = DistributedSleipnerDataset3D(P_x, val_idx, client, container, data_path, shape,
        normalize=True)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=nb, shuffle=False)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=nb, shuffle=False)
P_world._comm.Barrier()

# FNO
pfno = ParallelFNO(
        P_x,
        [nb, 3, *shape[:-1], 1],
        shape[-1],
        width,
        modes,
        device=device,
        dtype=dtype
)

# Training
num_epochs = 100
checkpoint_interval = 999
out_dir = '/home/azureuser/dfno/pfno/data'
parameters = [p for p in pfno.parameters()]
#criterion = distdl.nn.DistributedMSELoss(P_x).to(device)
criterion = DistributedRelativeLpLoss(P_x).to(device)
if len(parameters) > 0:
    optimizer = torch.optim.Adam(parameters, lr=1e-2, weight_decay=1e-3)
else:
    optimizer = None

# Keep track of loss history
if P_root.active:
    train_accs = []
    valid_accs = []

with cupy.cuda.Device(device_ordinal):
    # Training loop
    for i in range(num_epochs):

        # Loop over training data
        pfno.train()
        train_loss = 0
        n_train_batch = 0

        for j, (x, y) in enumerate(train_loader):

            t0 = time.time()
            x = x.to(device)
            y = y.to(device)

            y_hat = pfno(x)
            loss = criterion(y_hat, y)
            if P_root.active:
                #print(f'epoch = {i}, batch = {j}, loss = {loss.item()}')
                train_loss += loss.item()
                n_train_batch += 1

            #print("Rank: ", P_x.rank, "; Loss = ", loss)
            loss.backward()
            if optimizer is not None:
                optimizer.step()

            P_x._comm.Barrier()
            t1 = time.time()
            print(f'epoch = {i}, batch = {j}, dt = {t1-t0}')

        if P_root.active:
            #print(f'epoch = {i}, train loss = {train_loss/n_train_batch}')
            train_accs.append(train_loss/n_train_batch)

        P_x._comm.Barrier()

        # Loop over validation data
        pfno.eval()
        valid_loss = 0
        n_valid_batch = 0

        for j, (x, y) in enumerate(valid_loader):
            with torch.no_grad():
                t0 = time.time()
                x = x.to(device)
                y = y.to(device)

                y_hat = pfno(x)
                loss = criterion(y_hat, y)
                if P_root.active:
                    #print(f'epoch = {i}, batch = {j}, test loss = {loss.item()}')
                    valid_loss += loss.item()

                idx = 30
                fig = plt.figure()
                ax = fig.add_subplot(131)
                ax.imshow(np.squeeze(x.detach().cpu().numpy()[0,0,:,idx,:,0]).T)
                ax = fig.add_subplot(132)
                ax.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,:,idx,:,-1]).T)
                ax = fig.add_subplot(133)
                ax.imshow(np.squeeze(y_hat.detach().cpu().numpy()[0,0,:,idx,:,-1]).T)

                plt.savefig(f'pred_{P_x.rank}.png')
                
                if P_root.active:
                    n_valid_batch += 1

        if P_root.active:
            print(f'epoch = {i}, train loss = {train_loss/n_train_batch:08f}, val loss = {valid_loss/n_valid_batch:08f}')
            valid_accs.append(valid_loss/n_valid_batch)

        if (i+1) % checkpoint_interval == 0:

            if P_root.active:
                lossname = 'loss_epoch_' + str(i) + '.h5'
                fid = h5py.File(os.path.join(out_dir, lossname), 'w')
                fid.create_dataset('train_loss', data=train_accs)
                fid.create_dataset('valid_loss', data=valid_accs)
                fid.close()
                #print(f'rank = {P_x.rank}, saved loss: {lossname}')

            model_path = os.path.join(out_dir, f'model_{j:04d}_{P_x.rank:04d}.pt')
            torch.save(pfno.state_dict(), model_path)
            #print(f'rank = {P_x.rank}, saved model: {model_path}')

    # Save after training
    model_path = os.path.join(out_dir, f'model_{P_x.rank:04d}.pt')
    torch.save(pfno.state_dict(), model_path)
    print(f'rank = {P_x.rank}, saved model after final iteration: {model_path}')

    if P_root.active:
        print('training finished.') 
