import distdl, torch, os, time, h5py
import numpy as np
import azure.storage.blob
from mpi4py import MPI
from pfno import ParallelFNO4d
from distdl.backends.mpi.partition import MPIPartition
from sleipner_dataset import DistributedSleipnerDataset3D

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
P_x = P_feat_base.create_cartesian_topology_partition((1,1,1,n,1,1))
P_y = P_feat_base.create_cartesian_topology_partition((1,1,n,1,1,1))

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
num_k = (12, 12, 12, 8)

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
val_idx = torch.linspace(num_train+1, num_train+num_valid, num_valid, dtype=torch.int32).long()
valid_data = DistributedSleipnerDataset3D(P_x, val_idx, client, container, data_path, shape,
        normalize=True)

# Dataloaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=nb, shuffle=False)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=nb, shuffle=False)
P_world._comm.Barrier()

# FNO
pfno = ParallelFNO4d(
    P_world, 
    P_root,
    P_x,
    P_y,
    channel_in,
    channel_hidden,
    channel_out,
    shape,
    num_k
).to(device)

# Training
num_epochs = 20
checkpoint_interval = 999
out_dir = '/datadrive/philipp'
parameters = [p for p in pfno.parameters()]
criterion = distdl.nn.DistributedMSELoss(P_x).to(device)
if len(parameters) > 0:
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)
else:
    optimizer = None

# Keep track of loss history
if P_root.active:
    train_accs = []
    valid_accs = []

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
        #print(f'epoch = {i}, batch = {j}, dt = {t1-t0}')

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