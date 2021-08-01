#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=hard200_K5_rt8_rs6_T6_Ns6
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=short_mdbf
#SBATCH --partition=short_mdbf
#SBATCH --time=1:59:00
#SBATCH --output=/cta/users/aosman/MuSPCES/simulations/log/simulation/hard200_K5_rt8_rs6_T6_Ns6.out
#SBATCH --mem=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
/cta/users/aosman/MuSPCES/simulations/simulation.sh -c hard200_K5 -s 6 -t 6 --r-subject 0.6 --r-time 0.8

