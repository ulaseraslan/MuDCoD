#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}log/cross_validation"
mkdir -p "${curr_dir}configurations"

case_name=( easy200 medium200 hard200 )
time_horizon=(3)
num_subject=(3)
alpha_values=( 0.01 0.03 0.07 0.1 0.15 0.2)
beta_values=( 0.02 0.05 0.1 0.15 )
r_time=(0.3)
r_subject=(0.3)
size_coef=3
case_msd=(1 3)

qos="short"
n_jobs=4

for class_name in ${class_dcbm[@]}; do
  for case_msd in ${cases_msd[@]}; do
    for th in ${time_horizon[@]}; do
      for num_sbj in ${num_subject[@]}; do
        for rt in ${r_time[@]}; do
          for rs in ${r_subject[@]}; do
            for alpha in ${alpha_values[@]}; do
              rs_decimal=$(echo "${rs}"| cut -d'.' -f 2)
              rt_decimal=$(echo "${rt}"| cut -d'.' -f 2)
              alpha_decimal=$(echo "${alpha}"| cut -d'.' -f 2)
              name="pisces_${class_name}_case${case_msd}_rt${rt_decimal}_rs${rs_decimal}_th${th}_ns${num_sbj}_alpha${alpha_decimal}"
              echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=1:59:00
#SBATCH --output=${curr_dir}log/cross_validation/${name}.out
#SBATCH --mem=12G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a

${curr_dir}python simulation.py --verbose=False --case-msd=${case_msd} --time-horizon=${th} --class-dcbm=${class_name} --r-time=${rt} --num-subjects=${ns} --r-subject=${rs} --n-jobs=${n_jobs} cv-pisces --alpha=${alpha} --size-coef=${size_coef}
"           >${curr_dir}"tmp_slurm_submission.sh"
            sbatch ${curr_dir}"tmp_slurm_submission.sh"
            for beta in ${beta_values[@]}; do
              beta_decimal=$(echo "${beta}"| cut -d'.' -f 2)
              name="muspces_${class_name}_case${case_msd}_rt${rt_decimal}_rs${rs_decimal}_th${th}_ns${num_sbj}_alpha${alpha_decimal}_beta${beta_decimal}"
              echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${name}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=4
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=1:59:00
#SBATCH --output=${curr_dir}log/cross_validation/${name}.out
#SBATCH --mem=12G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a

${curr_dir}python simulation.py --verbose=False --case-msd=${case_msd} --time-horizon=${th} --class-dcbm=${class_name} --r-time=${rt} --num-subjects=${ns} --r-subject=${rs} --n-jobs=${n_jobs} cv-muspces --alpha=${alpha} --beta=${beta} --size-coef=${size_coef}
"             >${curr_dir}"tmp_slurm_submission.sh"
              sbatch ${curr_dir}"tmp_slurm_submission.sh"
              done
            done
          done
        done
      done
    done
  done
done
