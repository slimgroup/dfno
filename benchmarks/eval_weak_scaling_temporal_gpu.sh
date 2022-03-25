#!/bin/bash
set -x

data_dir=eval_weak_scaling_temporal_gpu
jsrun -n 1 --cpu_per_rs=7 --rs_per_host=1 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 4 -ps 1 1 1 1 1 1 -w 20 -nt 16 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 2 --cpu_per_rs=7 --rs_per_host=2 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 8 -ps 1 1 2 1 1 1 -w 20 -nt 32 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 4 --cpu_per_rs=7 --rs_per_host=4 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 16 -ps 1 1 2 2 1 1 -w 20 -nt 64 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 6 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 24 -ps 1 1 3 2 1 1 -w 20 -nt 96 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 12 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 48 -ps 1 1 3 2 2 1 -w 20 -nt 192 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 24 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 96 -ps 1 1 4 3 2 1 -w 20 -nt 384 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 48 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 192 -ps 1 1 4 4 3 1 -w 20 -nt 768 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 96 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 384 -ps 1 1 6 4 4 1 -w 20 -nt 1536 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 192 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 768 -ps 1 1 8 6 4 1 -w 20 -nt 3072 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 384 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 1536 -ps 1 1 8 8 6 1 -w 20 -nt 6144 -d cuda -ngpu 1 -bt eval -o $data_dir
jsrun -n 768 --cpu_per_rs=7 --rs_per_host=6 --tasks_per_rs=1 --gpu_per_rs=1 --bind=rs python3 bench.py -is 1 1 64 64 64 1 -m 4 4 4 3072 -ps 1 1 12 8 8 1 -w 20 -nt 12288 -d cuda -ngpu 1 -bt eval -o $data_dir
