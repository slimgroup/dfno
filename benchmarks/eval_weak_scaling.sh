#!/bin/bash

data_dir=eval_weak_scaling
mpirun -np 1  --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 32  32  32  1 -ps 1 1 1 1 1 1 -w 20 -m 4  4  4  4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 2  --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64  32  32  1 -ps 1 1 2 1 1 1 -w 20 -m 8  4  4  4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 4  --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64  64  32  1 -ps 1 1 2 2 1 1 -w 20 -m 8  8  4  4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 8  --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 64  64  64  1 -ps 1 1 2 2 2 1 -w 20 -m 8  8  8  4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 16 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 128 64  64  1 -ps 1 1 4 2 2 1 -w 20 -m 16 8  8  4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 32 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 128 128 64  1 -ps 1 1 4 4 2 1 -w 20 -m 16 16 8  4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 48 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 128 128 96  1 -ps 1 1 4 4 3 1 -w 20 -m 16 16 12 4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 64 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 128 128 128 1 -ps 1 1 4 4 4 1 -w 20 -m 16 16 16 4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 80 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 160 128 128 1 -ps 1 1 5 4 4 1 -w 20 -m 20 16 16 4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
mpirun -np 96 --bind-to core --map-by numa:PE=1 python3 bench.py -is 1 1 192 128 128 1 -ps 1 1 6 4 4 1 -w 20 -m 24 16 16 4 -nt 10 -d cpu -ngpu 0 -bt eval -o $data_dir
