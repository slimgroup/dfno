import subprocess as sp
import os
import sched, time

# https://stackoverflow.com/questions/67707828/how-to-get-every-seconds-gpu-usage-in-python
def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values

def profile_gpu_memory(outfile, dt=1.0):
    while True:
        with open(outfile, 'a') as f:
            muv = get_gpu_memory()
            for i, m in enumerate(muv):
                print(m)
                f.write(str(m))
                if i < len(muv)-1:
                    f.write(', ')
            f.write('\n')
        time.sleep(dt)
