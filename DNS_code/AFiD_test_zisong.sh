#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob_hybrid.out.%j
#SBATCH -e ./tjob_hybrid.err.%j
# Initial working directory:
#SBATCH -D ./
# Job Name:
#SBATCH -J AFiD_test_zisong
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
# Enable Hyperthreading:
#SBATCH --ntasks-per-core=1
#
# Request 180 GB of main memory per node in units of MB:
#SBATCH --mem=18500
#
#SBATCH --mail-type=none
#SBATCH --mail-user=zhouz@mps.mpg.de
#
# Wall clock Limit:
#SBATCH --time=24:00:00

# Load compiler and MPI modules with explicit version specifications,
# consistently with the versions used to build the executable.
module load intel/21.3.0 mkl/2021.3 impi/2021.3 fftw-mpi/3.3.10 hdf5-mpi/1.12.1

# enable over-subscription of physical cores by MPI ranks
export PSM2_MULTI_EP=0

# Run the program:
srun ./afid