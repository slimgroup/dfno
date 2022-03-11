#!/bin/bash

data_dir=grad_weak_scaling_spatial_gpu
mpirun -np 1 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 4 -ps 1 1 1 1 1 1 -w 20 -nt 10 -d cuda -ngpu 6 -bt grad -o $data_dir
mpirun -np 2 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 128 64 64 1 -m 8 4 4 4 -ps 1 1 2 1 1 1 -w 20 -nt 10 -d cuda -ngpu 6 -bt grad -o $data_dir
mpirun -np 4 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 128 128 64 1 -m 8 8 4 4 -ps 1 1 2 2 1 1 -w 20 -nt 10 -d cuda -ngpu 6 -bt grad -o $data_dir
