#!/bin/bash
set -x

data_dir=grad_weak_scaling_spatial_gpu
jsrun -n 1 --cpu_per_rs=7 --rs_per_host=1 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 4 -ps 1 1 1 1 1 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 2 --cpu_per_rs=7 --rs_per_host=2 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 128 64 64 1 -m 8 4 4 4 -ps 1 1 2 1 1 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 4 --cpu_per_rs=7 --rs_per_host=4 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 128 128 64 1 -m 8 8 4 4 -ps 1 1 2 2 1 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 6 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 192 128 64 1 -m 12 8 4 4 -ps 1 1 3 2 1 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 12 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 192 128 128 1 -m 12 8 8 4 -ps 1 1 3 2 2 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 24 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 256 192 128 1 -m 16 12 8 4 -ps 1 1 4 3 2 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 48 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 256 256 192 1 -m 16 16 12 4 -ps 1 1 4 4 3 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 96 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 384 256 256 1 -m 24 16 16 4 -ps 1 1 6 4 4 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 192 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 512 384 256 1 -m 32 24 16 4 -ps 1 1 8 6 4 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 384 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 512 512 384 1 -m 32 32 24 4 -ps 1 1 8 8 6 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
jsrun -n 768 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 768 512 512 1 -m 48 32 32 4 -ps 1 1 12 8 8 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt grad -o $data_dir
