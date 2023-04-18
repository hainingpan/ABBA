#!/bin/bash

# Run the Python script with nu=[12,6]
python sweep.py --nu 12 6 -t1 -1 -1 1 -t2 -0.3 -0.5 11 -U0 4 8 9 -U1 0 0 1 -H 0.6 -n 33 --thres 1e-10

# Run the Python script with nu=[18,9]
python sweep.py --nu 18 9 -t1 -1 -1 1 -t2 -0.3 -0.5 11 -U0 4 8 9 -U1 0 0 1 -H 0.6 -n 33 --thres 1e-10

t2 0 0.5, U0start 1