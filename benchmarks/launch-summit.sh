#!/bin/bash

module load open-ce/1.5.0-py39-0
module unload xl
conda activate $MYWORK/test-env
export CUPY_CACHE_DIR=$MYWORK/.cupy/kernel_cache
cd $MYWORK/dfno/pfno

# Uncomment this line to run with cuda-aware MPI support
# export CUDA_AWARE=1

# Uncomment this line to run with GPU support, but without cuda-aware MPI (default)
export USE_CUDA=1

# Leaving both of the above lines commented will run benchmarking scripts on
# CPU device

ulimit -c 0
./eval_weak_scaling_spatial_gpu.sh $1
./eval_weak_scaling_temporal_gpu.sh $1
./grad_weak_scaling_spatial_gpu.sh $1
./grad_weak_scaling_temporal_gpu.sh $1