#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}simulation_log"
mkdir -p "${curr_dir}results"

case_name=( )
time_horizon=(2 4 6)
num_subject=(1 2 4 6)
r_time=(0.001 0.2 0.5)
r_subject=(0.001 0.2 0.5)

qos="mid"

for case in ${case_name[@]}; do
    for th in ${time_horizon[@]}; do
        for num_sbj in ${num_subject[@]}; do
            for rt in ${r_time[@]}; do
                for rs in ${r_subject[@]}; do
                    name="${case}_rt${rt}_rs${rs}_T${th}_Ns${num_sbj}"
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
${curr_dir}simulation.sh -c ${case} -s ${num_sbj} -t ${th} --r-subject ${rs} --r-time ${rt}
" > ${curr_dir}"tmp_slurm_submission.sh"
                    sbatch ${curr_dir}"tmp_slurm_submission.sh"
                done
            done
        done
    done
done
