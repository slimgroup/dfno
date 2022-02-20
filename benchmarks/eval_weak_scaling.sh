#!/bin/bash

data_dir=eval_weak_scaling

mpirun -np 1 python bench.py -is 1 1 32 32 32 1    -ps 1 1 1 1 1 1 -w 20 -m 4 4 4 4    -nt 32 -d cpu -ngpu 1 -bt eval -o $data_dir
mpirun -np 2 python bench.py -is 1 1 64 32 32 1    -ps 1 1 1 1 1 1 -w 20 -m 8 4 4 4    -nt 32 -d cpu -ngpu 2 -bt eval -o $data_dir
mpirun -np 3 python bench.py -is 1 1 64 64 32 1    -ps 1 1 1 1 1 1 -w 20 -m 8 8 4 4    -nt 32 -d cpu -ngpu 3 -bt eval -o $data_dir
mpirun -np 4 python bench.py -is 1 1 64 64 64 1    -ps 1 1 1 1 1 1 -w 20 -m 8 8 8 4    -nt 32 -d cpu -ngpu 4 -bt eval -o $data_dir
mpirun -np 5 python bench.py -is 1 1 128 64 64 1   -ps 1 1 1 1 1 1 -w 20 -m 8 8 8 8    -nt 32 -d cpu -ngpu 5 -bt eval -o $data_dir
mpirun -np 6 python bench.py -is 1 1 128 128 64 1  -ps 1 1 1 1 1 1 -w 20 -m 16 8 8 8   -nt 32 -d cpu -ngpu 6 -bt eval -o $data_dir
mpirun -np 7 python bench.py -is 1 1 128 128 128 1 -ps 1 1 1 1 1 1 -w 20 -m 16 16 8 8  -nt 32 -d cpu -ngpu 7 -bt eval -o $data_dir
mpirun -np 8 python bench.py -is 1 1 256 128 128 1 -ps 1 1 1 1 1 1 -w 20 -m 16 16 16 8 -nt 32 -d cpu -ngpu 8 -bt eval -o $data_dir
