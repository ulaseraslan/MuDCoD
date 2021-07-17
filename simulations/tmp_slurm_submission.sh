#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=test10K2
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=11:59:00
#SBATCH --output=/home/bo/Files/projects/MuSPCES/simulations/cv_log/test10K2.out
#SBATCH --mem=16G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python -W ignore /home/bo/Files/projects/MuSPCES/simulations/cv_dcbm.py --case test10K2

