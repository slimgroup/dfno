#!/bin/bash
outdir=$1
shift
if [[ ! -d $PROFILE ]]; then
  python3 bench.py "$@"
  exit 0
fi

if [[ $OMPI_COMM_WORLD_RANK == $(( $OMPI_COMM_WORLD_SIZE - 1)) || $OMPI_COMM_WORLD_RANK == 0 ]]; then
  nsys profile --force-overwrite=true -o ${outdir}/${outdir}_%q{OMPI_COMM_WORLD_RANK}_%q{OMPI_COMM_WORLD_SIZE}.qdrep python3 bench.py "$@" --mydummyargument
else
  python3 bench.py "$@"
fi
