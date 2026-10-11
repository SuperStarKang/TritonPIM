if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <program.py> [args...]" >&2
  exit 1
fi

exec sudo -E env "PATH=$PATH" "HOME=$HOME" "PYTHONPATH=$PYTHONPATH" UPMEM_BACKEND=hw LD_LIBRARY_PATH=/home/dlrkdals/.local/lib/upmem-override:$LD_LIBRARY_PATH python "$@" 