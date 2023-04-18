#!/bin/bash
#SBATCH -n 32
#SBATCH -t 10:00:00 
#SBATCH --mem=64G 
#SBATCH -p kim

# python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 -0.3 -0.5 11 -U0 4 8 9 -U1 0 0 1 -H 0.6 -n 33 --thres 1e-10
# python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 -0.2 -0.2 1 -U0 6 10 5 -U1 1 10 10 -H 0.6 -n 33 --thres 1e-10
# python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 -0.2 -0.2 1 -U0 1 10 10 -U1 1 10 10 -H 0.3 -n 33 --thres 1e-10
# python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 -0.0 -0.0 1 -U0 1 10 10 -U1 1 10 10 -H 0.6 -n 33 --thres 1e-10

# python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 -0.2 -0.2 1 -U0 5 5 1 -U1 .5 3.5 16 -H 0.6 -n 33 --thres 1e-10
python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 0 0 1 -U0 0.1 1 5 -U1 0 0 1 -H 0.4 -n 198 --thres 1e-10
