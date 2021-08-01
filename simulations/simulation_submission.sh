#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}log/simulation"
mkdir -p "${curr_dir}../results/simulation_outputs"

case_name=( medium200_K5 hard200_K5 )
time_horizon=( 2 4 6 )
num_subject=(2 4 6)
r_time=(0.4 0.6 0.8)
r_subject=(0.2 0.4 0.6)

qos="short"
n_jobs="short"

for case in ${case_name[@]}; do
  for th in ${time_horizon[@]}; do
    for num_sbj in ${num_subject[@]}; do
      for rt in ${r_time[@]}; do
        for rs in ${r_subject[@]}; do
          rs_decimal=$(echo "${rs}"| cut -d'.' -f 2)
          rt_decimal=$(echo "${rt}"| cut -d'.' -f 2)
          name="${case}_rt${rt_decimal}_rs${rs_decimal}_T${th}_Ns${num_sbj}"
          echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=1:59:00
#SBATCH --output=${curr_dir}log/simulation/${name}.out
#SBATCH --mem=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
${curr_dir}simulation.sh -c ${case} -s ${num_sbj} -t ${th} --r-subject ${rs} --r-time ${rt}
"         > ${curr_dir}"tmp_slurm_submission.sh"
          sbatch ${curr_dir}"tmp_slurm_submission.sh"
        done
      done
    done
  done
done
