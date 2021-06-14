#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"


simulation_name=(bh250r0K7 bh250r2K7 bh250r5K7 by100r0K3 by100r2K3 by100r5K3 dp250r0K5 dp250r2K5 dp250r5K5 lr250r0K5 lr250r2K5 lr250r5K5 pt100r0K2 pt100r2K2 pt100r5K2 qs100r0K2 qs100r2K2 qs100r5K2 zk250r0K5 zk250r2K5 zk250r5K5)
time_horizon=(2 4 6)
num_subject=(1 2 4 6)

qos="mid"

for sname in ${simulation_name[@]}; do
    for th in ${time_horizon[@]}; do
        for ns in ${num_subject[@]}; do
            name="${sname}_T${th}_Ns${ns}"
            echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=11:59:00
#SBATCH --output=${curr_dir}simulation_log/${name}.out
#SBATCH --mem=16G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
${curr_dir}simulation.sh -n ${sname} -t ${th} -s ${ns}
    "       >${curr_dir}"slurm_submission.sh"
            sbatch ${curr_dir}"slurm_submission.sh"
        done
    done
done
