#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}log/"
mkdir -p "${curr_dir}../results"

classes_dcbm=(easy100 medium100 hard100 easy500 medium500 hard500)
time_horizon=(2 4 8)
num_subject=(1 4 16)
r_time=(0.0 0.2 0.5)
r_subject=(0.0 0.2 0.5)
scenarios_msd=(1 3)

qos="mid"
time="23:59:00"

for class_dcbm in ${classes_dcbm[@]}; do
  for th in ${time_horizon[@]}; do
    for ns in ${num_subject[@]}; do
      for rt in ${r_time[@]}; do
        for rs in ${r_subject[@]}; do
          for scenario_msd in ${scenarios_msd[@]}; do
            rs_dec=$(echo "${rs}"| cut -d'.' -f 2)
            rt_dec=$(echo "${rt}"| cut -d'.' -f 2)
            name="${class_dcbm}_scenario${scenario_msd}_th${th}_rt${rt_dec}_ns${ns}_rs${rs_dec}"
            echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=1
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=${time}
#SBATCH --output=${curr_dir}log/${name}.out
#SBATCH --mem=8G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
${curr_dir}run_simulation.sh \
  --class-dcbm ${class_dcbm} --scenario-msd ${scenario_msd} \
  --time-horizon ${th} --r-time ${rt} \
  --num-subjects ${ns} --r-subject ${rs}
"           > ${curr_dir}"tmp_slurm_submission.sh"
            sbatch ${curr_dir}"tmp_slurm_submission.sh"
          done
        done
      done
    done
  done
done
