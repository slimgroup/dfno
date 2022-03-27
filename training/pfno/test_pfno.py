import distdl, torch, os, time, h5py
import numpy as np
import azure.storage.blob
from mpi4py import MPI
from pfno import ParallelFNO4d
from distdl.backends.mpi.partition import MPIPartition
from sleipner_dataset import DistributedSleipnerDataset3D
from distdl.nn.repartition import Repartition

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

# Collectors
collect_x = Repartition(P_x, P_root)
collect_y = Repartition(P_x, P_root)
collect_y_ = Repartition(P_x, P_root)

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
val_idx = torch.linspace(num_train, num_train+num_valid, num_valid, dtype=torch.int32).long()
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
)

# Load trained network
out_dir = '/datadrive/philipp'
model_path = os.path.join(out_dir, f'model_{P_x.rank:04d}.pt')
pfno.load_state_dict(torch.load(model_path))
pfno = pfno.to(device)
pfno.eval()

# Get sample
x, y = next(iter(valid_loader))
x = x.to(device)
y = y.to(device)

# Predict
with torch.no_grad():
    y_  = pfno(x)

# Collect on root
x = collect_x(x)
y = collect_y(y)
y_ = collect_y_(y_)

# Save result
if P_root.active:
    fid = h5py.File(os.path.join(out_dir, 'fno_sample.h5'), 'w')
    fid.create_dataset('x', data=x.detach().cpu())
    fid.create_dataset('y_', data=y_.detach().cpu())
    fid.create_dataset('y', data=y.detach().cpu())
    print("Saved data sample!")