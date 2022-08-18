#!/bin/bash

if [[ -z $MYWORK ]]; then
	echo "Please set variable MYWORK to somewhere writeable by the GPU nodes."
	echo "For example:"
	echo "export MYWORK=/gpfs/alpine/scratch/rishi/csc471/work"
	exit 0
fi

if [[ ! -d $MYWORK ]];  then
  echo "Please create the work directory:$MYWORK"
fi

cd $MYWORK
if [[ ! -d $MYWORK/dfno ]]; then
  echo "Copying dfno files"
  cp -r /gpfs/alpine/csc471/world-shared/dfno $MYWORK
fi

if [[ ! -d $MYWORK/distdl ]]; then
  echo "Copying distdl files"
  cp -r /gpfs/alpine/csc471/world-shared/distdl $MYWORK
fi

echo "Loading modules"
module load open-ce/1.5.0-py39-0
module unload xl

if [[ ! -d $MYWORK/test-env ]]; then
  echo "Creating conda environment. This may take a while"
  conda create --clone open-ce-1.5.0-py39-0 -p $MYWORK/test-env
fi

echo "Activating conda environment"
conda activate $MYWORK/test-env

echo "Installing mpi4py"
#install mpi4py
pip3 show -q mpi4py || MPICC="mpicc -shared" pip3 install --no-binary=mpi4py mpi4py

echo "Installing nvtx"
#install nvtx
pip3 show -q nvtx || pip3 install nvtx

echo "Installing cupy. This may take a while"
#install cupy
pip3 show -q cupy || pip3 install cupy

echo "Installing dfno"
#install dfno
cd dfno
pip3 show -q dfno || pip3 install -e .
cd ..

echo "Re-Installing distdl from local"
#install custom distdl over base
cd distdl
#pip3 show -q distdl || pip3 install -e .
pip3 install -e .
cd ..

echo "Fixing paths in launch script"
sed -i "s|MYWORK|$MYWORK|g" dfno/pfno/launch-summit.sh

pip3 show -q dfno
if [[ $? -eq 0 ]]; then
 echo "dfno was installed properly"
else
 echo "dfno was not installed properly"
fi
conda deactivate