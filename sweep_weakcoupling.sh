#!/bin/bash
#SBATCH -n 8
#SBATCH -t 2:00:00 
#SBATCH --mem=64G 
#SBATCH -p kim
# python sweep.py --nu 12 6 -t1 -1 -1 1 -t2 0 0 1 -U0 0.1 1 10 -U1 0 0 1 -H 0.6 -n 69 --no-hartree --thres 1e-10
# python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 0 0 1 -U0 0 0 1 -U1 0.1 1 10 -H 0.3 -n 69 --no-hartree --thres 1e-10
python sweep.py --nu 14 7 -t1 -1 -1 1 -t2 0 0 1 -U0 0 0 1 -U1 0.1 1 21 -H 0.5 -n 69 --no-hartree --thres 1e-5
# python sweep2.py --nu 18 9 -t1 -1 -1 1 -t2 -0.75 -0.75 1 -U0  5 5 1 -U1 0 0 1 -H 0.6 -n 108 --thres 1e-10