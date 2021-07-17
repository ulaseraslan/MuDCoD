#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -c|--case)
    case_name="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--time-horizon)
    time_horizon="$2"
    shift
    shift
    ;;
    -s|--num-subject)
    num_subject="$2"
    shift
    shift
    ;;
    --r-subject)
    r_subject="$2"
    shift
    shift
    ;;
    --r-time)
    r_time="$2"
    shift
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done

curr_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/"

echo "Name: ${case_name}"
echo "Number of subject:${num_subject}"
echo "Time horizon: ${time_horizon}"

num_simulation=100

for ((i = 0 ; i < num_simulation ; i++)); do
    python -W ignore "${curr_dir}multisubject_dynamic_dcbm.py" --case "${case_name}" --identity "${i}" --num-subject "${num_subject}" --time-horizon "${time_horizon}" --r-subject "${r_subject}" --r-time "${r_time}"
done
