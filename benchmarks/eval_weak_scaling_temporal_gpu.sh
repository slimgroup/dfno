#!/bin/bash

data_dir=eval_weak_scaling_temporal_gpu
mpirun -np 1 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 4 -ps 1 1 1 1 1 1 -w 20 -nt 10 -d cuda -ngpu 6 -bt eval -o $data_dir
mpirun -np 2 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 8 -ps 1 1 1 1 1 2 -w 20 -nt 20 -d cuda -ngpu 6 -bt eval -o $data_dir
mpirun -np 4 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 16 -ps 1 1 1 1 1 4 -w 20 -nt 40 -d cuda -ngpu 6 -bt eval -o $data_dir
