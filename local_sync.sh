SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
echo ${SCRIPTPATH}

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:

wrt="--dry-run"
del=""

while getopts "wd" opt; do
  case "$opt" in
    w)  wrt=""
      ;;
    d)  del="--delete"
      ;;
  esac
done

shift $((OPTIND-1))

[ "${1:-}" = "--" ] && shift

path_list=( "/dypoces/" "/simulations/")
for sub_path in "${path_list[@]}"
do
    echo ${sub_path}
    SYNC=tosun_away:dypoces${sub_path}
    rsync -arviz --exclude="**/log/*" $SYNC ${SCRIPTPATH}${sub_path}  ${wrt} ${del}
done
