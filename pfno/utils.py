import subprocess as sp
import os
import sched, time
import torch

from contextlib import nullcontext

# https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        tmp = sp.check_output(COMMAND,stderr=sp.STDOUT,shell=True)
        memory_use_info = output_to_list(tmp)[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

def profile_gpu_memory(outfile, dt=1.0):
    t0 = time.time()
    with open(outfile, 'w') as f:
        while True:
            muv = get_gpu_memory()
            f.write(f'{str(time.time()-t0)}, ')
            for i, m in enumerate(muv):
                f.write(str(m))
                if i < len(muv)-1:
                    f.write(', ')
            f.write('\n')
            f.flush()
            time.sleep(dt)

def get_env(P, num_gpus=1):

    cuda_aware = 'CUDA_AWARE' in os.environ
    use_cuda = 'USE_CUDA' in os.environ or cuda_aware
    device_ordinal = P.rank % num_gpus
    device = torch.device(f'cuda:{device_ordinal}') if use_cuda else torch.device('cpu')
    
    if cuda_aware:
        import cupy
        ctx = cupy.cuda.Device(device_ordinal)
    else:
        ctx = nullcontext()

    return use_cuda, cuda_aware, device_ordinal, device, ctx
