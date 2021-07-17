#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}cv_log"
mkdir -p "${curr_dir}simulation_configs"

case_name=(easy200K5 medium200K5 hard200K5)
time_horizon=(4)
num_subject=(4)
r_time=(0.2)
r_subject=(0.2)
size_coef=3

qos="mid"

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
#SBATCH --ntasks-per-node=8
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=11:59:00
#SBATCH --output=${curr_dir}cv_log/${name}.out
#SBATCH --mem=16G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python -W ignore "${curr_dir}cv_dcbm.py" -c "${case}" -s "${num_sbj}" -t "${th}" --r-subject "${rs}" --r-time "${rt}" --size-coef "${size_coef}"
"    >${curr_dir}"tmp_slurm_submission.sh"
                    sbatch ${curr_dir}"tmp_slurm_submission.sh"
                done
            done
        done
    done
done
