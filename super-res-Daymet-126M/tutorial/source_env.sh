#!/bin/bash

module load PrgEnv-gnu
module load gcc/12.2.0
module load rocm/5.7.0 libtool

eval "$(/lustre/orion/world-shared/stf218/atsaris/env_test_march/miniconda/bin/conda shell.bash hook)"
#conda activate /lustre/orion/lrn036/world-shared/patrickfan/torch-sr
conda activate /lustre/orion/lrn036/world-shared/patrickfan/torch-v25

export LD_LIBRARY_PATH=/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl/build:/lustre/orion/world-shared/stf218/atsaris/env_test_march/rccl-plugin-rocm570/lib/:/opt/cray/libfabric/1.15.2.0/lib64/:/opt/rocm-5.7.0/lib:$LD_LIBRARY_PATH
