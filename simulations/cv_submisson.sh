#!/bin/bash

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"
mkdir -p "${curr_dir}cv_log"
mkdir -p "${curr_dir}simulation_configs"

case_name=( )

qos="mid"

for case in ${case_name[@]}; do
    echo "#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=${case}
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=8
#SBATCH --qos=${qos}_mdbf
#SBATCH --partition=${qos}_mdbf
#SBATCH --time=11:59:00
#SBATCH --output=${curr_dir}cv_log/${case}.out
#SBATCH --mem=16G

# Set stack size to unlimited

ulimit -s unlimited
ulimit -l unlimited
ulimit -a
python -W ignore "${curr_dir}cv_dcbm.py" --case "${case}"
"    >${curr_dir}"tmp_slurm_submission.sh"
    sbatch ${curr_dir}"tmp_slurm_submission.sh"
done
