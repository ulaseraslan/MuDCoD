#!/bin/bash
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -n|--name)
    simulation_name="$2"
    shift # past argument
    shift # past value
    ;;
    -t|--time)
    time_horizon="$2"
    shift # past argument
    shift #past value
    ;;
    -s|--subject)
    num_subject="$2"
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

echo "Name: ${simulation_name}"
echo "Number of subject:${num_subject}"
echo "Time horizon: ${time_horizon}"

num_simulaiton=100

## python -W ignore "${curr_dir}cv_dcbm.py" "${simulation_name}" "${time_horizon}" "${num_subject}"
for ((i = 0 ; i < num_simulaiton ; i++)); do
    python -W ignore "${curr_dir}multisubject_dynamic_dcbm.py" "${simulation_name}_T${time_horizon}_Ns${num_subject}" "${i}"
done

