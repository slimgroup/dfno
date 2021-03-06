import torch, os, h5py
import numpy as np
import azure.storage.blob
import matplotlib.pyplot as plt

from dfno import DistributedFNO, create_standard_partitions, get_env
from sleipner_dataset import DistributedSleipnerDataset3D
from distdl.nn.repartition import Repartition
from dotenv import load_dotenv

# Load dataset information from environment
load_dotenv()

# Partitions
n = 4
P_world, P_x, P_root = create_standard_partitions((1, 1, 1, n, 1, 1))
use_cuda, cuda_aware, device_ordinal, device, ctx = get_env(P_x, num_gpus=n)
dtype = torch.float32

# Collectors
collect_x = Repartition(P_x, P_root)
collect_y = Repartition(P_x, P_root)
collect_y_ = Repartition(P_x, P_root)

# Reproducibility
torch.manual_seed(P_x.rank + 123)
np.random.seed(P_x.rank + 123)

# Data dimensions
nb = 1
shape = (60, 60, 64, 30)    # X Y Z T
num_train = 1
num_valid = 1

# Network dimensions
channel_in = 2
width = 20
channel_out = 1
modes = (12, 12, 12, 8)

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
dfno = DistributedFNO(
        P_x,
        [nb, 3, *shape[:-1], 1],
        shape[-1],
        width,
        modes,
        device=device,
        dtype=dtype
)

# Load trained network
out_dir = 'data/'
model_path = os.path.join(out_dir, f'model_{P_x.rank:04d}.pt')
dfno.load_state_dict(torch.load(model_path))
dfno.eval()

# Get sample
x, y = next(iter(valid_loader))
x = x.to(device)
y = y.to(device)

print(x.shape, y.shape)

# Predict
with ctx:
    with torch.no_grad():
        y_ = dfno(x)

    # Collect on root
    x = collect_x(x)
    y = collect_y(y)
    y_ = collect_y_(y_)

    # Save result
    if P_root.active:
        
        idx = 30
        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(np.squeeze(x.detach().cpu().numpy()[0,0,:,idx,:,0]).T)
        ax = fig.add_subplot(132)
        ax.imshow(np.squeeze(y.detach().cpu().numpy()[0,0,:,idx,:,-1]).T)
        ax = fig.add_subplot(133)
        ax.imshow(np.squeeze(y_.detach().cpu().numpy()[0,0,:,idx,:,-1]).T)

        plt.savefig('pred.png')

        fid = h5py.File(os.path.join(out_dir, 'fno_sample.h5'), 'w')
        fid.create_dataset('x', data=x.detach().cpu())
        fid.create_dataset('y_', data=y_.detach().cpu())
        fid.create_dataset('y', data=y.detach().cpu())
        print("Saved data sample!")
