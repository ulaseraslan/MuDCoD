#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=zk250r5K5_T6_Ns6
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=mid_mdbf
#SBATCH --partition=mid_mdbf
#SBATCH --time=11:59:00
#SBATCH --output=/cta/users/aosman/PisCES-multisubject/simulations/simulation_log/zk250r5K5_T6_Ns6.out
#SBATCH --mem=16G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
/cta/users/aosman/PisCES-multisubject/simulations/simulation.sh -n zk250r5K5 -t 6 -s 6
    
