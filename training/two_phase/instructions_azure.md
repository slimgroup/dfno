# Instructions for Running Two-Phase Flow Example on Azure

First, create a virtual machine from the CLI using

```
az vm create \
  --resource-group <group name> \
  --name <vm name> \
  --size Standard_NC24s_v3 \
  --image nvidia:nvidia_hpc_sdk_vmi:nvidia_hpc_sdk_vmi_22_1_0:22.01.0 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --location <location>
``` 

SSH into this machine. Then, install pip3 via

```
sudo apt install python3-pip
```

Install initial dependencies with

```
CFLAGS=-noswitcherror pip3 install torch numpy mpi4py
```

Next, clone the repository via

```
git clone https://github.com/slimgroup/dfno
```

Inside of the cloned directory, install in editable mode using

```
pip3 install -e .
```

Navigate to the `training/two_phase` directory, and install extra dependencies
with

```
pip3 install -r requirements_two_phase.txt
```

Given that you have access to a sleipner dataset with the correct setup, create
a `.env` file with the following values inside of it:

```
CONTAINER=<container name>
DATA_PATH=<path to data in container>
ACCOUNT_URL=<azure blob storage account url>
SLEIPNER_CREDENTIALS=<storage credentials>
```

Assuming all of these steps have been completed correctly, you may now train the
network via

```
CUDA_AWARE=1 mpirun -np 4 python3 train_two_phase.py
```

which will output a checkpointed model every 10 epochs. If you wish to generate
a sample from any of these models, rename each rank's model called
`model_<epoch>_<rank>.pt` to `model_<rank>.pt` and then run

```
CUDA_AWARE=1 mpirun -np 4 python3 test_two_phase.py
```
