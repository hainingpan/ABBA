#!/bin/bash
#SBATCH -J jupyter_file             # Job name
#SBATCH -o jupyter_file_%j.out         # output file (%j expands to jobID)
#SBATCH -e jupyter_file_%j.err         # error log file (%j expands to jobID)
#SBATCH -N 1                 # Total number of nodes requested
#SBATCH -n 8                 # Total number of cores requested
#SBATCH --get-user-env            # retrieve the users login environment
p             # server memory requested (per node)
#SBATCH -t 6:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=default_partition    # Request partition
export PATH=/home/hp364/.pyenv/versions/anaconda3-2022.05/bin:$PATH
XDG_RUNTIME_DIR=/tmp/hp364 jupyter-notebook --ip=0.0.0.0 --port=8891