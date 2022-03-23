import subprocess as sp
import os
import sched, time

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def profile_gpu_memory(outfile, dt=1.0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    with open(outfile, 'w') as f:
        while True:
            info = nvmlDeviceGetMemoryInfo(handle)
            f.write(f'{info.total}, {info.free}, {info.used}\n')
            f.flush()
            time.sleep(dt)
