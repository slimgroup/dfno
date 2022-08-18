# Set the GPU billing account for your project

NOTE: THIS IS ***MY*** ACCOUNT ID. Use your own!! The rest of the instructions won't work without it!

```sh
export ACCOUNT=m3863_g
```

# Activate the conda environment

```sh
module load python
conda activate /global/cfs/cdirs/m3863/mark/conda-python-3.9-mpi-cuda
```

## Recreating the conda environment

If you cannot access the above environment, here is how you can make your own.

```sh
module load python
conda create --prefix $CFS/YOUR_ACCOUNT/YOUR_CONDA_FOLDER python=3.9 -y
conda activate $CFS/YOUR_ACCOUNT/YOUR_CONDA_FOLDER
conda install cudatoolkit=11.3.1
# use specific prebuilt version of torch for this version of cuda
pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
# use specific prebuilt version of cupy for this version of cuda
pip3 install cupy_cuda113
# mpi4py build needs a little setup
MPICC="cc -target-accel=nvidia80 -shared" pip3 install --force --no-cache-dir --no-binary=mpi4py mpi4py
# install distdl from tgrady's cuda-aware distdl branch
pip3 install git+https://github.com/thomasjgrady/distdl@cuda-aware-2
# install other dfno requirements
pip3 install mat73 matplotlib numpy scipy
```

Note, DFNO's `setup.py` file defines the current set of requirements, see
there for updates.

# Change to the correct directory

```sh
cd `.../dfno/benchmarks`
```

# Generate script files

```sh
python3 gen_scripts.py --system=perlmutter
```

# Launch a little job to test it

```sh
salloc --account=$ACCOUNT --constraint=gpu --nodes=1 --qos=regular --time=0:30:00 --job-name=eval_weak_scaling_spatial_gpu.4 eval_weak_scaling_spatial_gpu.sh 4
sbatch --account=$ACCOUNT --constraint=gpu --nodes=1 --qos=regular --time=0:30:00 --job-name=eval_weak_scaling_spatial_gpu.4 eval_weak_scaling_spatial_gpu.sh 4
```

Watch them run and make sure they complete successfully.

The `salloc` command runs the job in the foreground, `sbatch` submits it as a
background job.  The sbatch command will return immediately; you can use
`squeue -u $USER` to see the status of it.  The background job will create a
log file like `slurm-$JOBID.out` in the current working directory when it
starts.

Both of these jobs need permission to write logs and benchmark results to the
directory you're running it in.

Note, if you forgot to set $ACCOUNT, you will see an error message like this:

```
salloc: error: Job submit/allocate failed: Invalid account or account/partition combination specified
```

Otherwise, you should eventually see output like:
```
initialize
fake eval
fake eval done
real eval
real eval done
```

# Submit the full set of jobs

```sh
./submit_perlmutter.sh
```
