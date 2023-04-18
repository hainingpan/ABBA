import argparse
import numpy as np
import pickle
from MF import *

def main(h, n, thres):
    U_list = np.linspace(0.1, 1, 10)
    gap_noH = []
    for U in U_list:
        print(U)
        params = Params(t=[0, -1, 0], U=[U, 0], h=h, nu=[12, 6], n=n)
        gap_list, energy_list, spin_list, ave = params.iteration(print_opt='', thres=thres)
        gap_noH.append(gap_list[-1])

    U_list_anal = np.linspace(1e-2, 1, 100)
    gap_anal = np.array([gap_vs_U(U, params) for U in U_list_anal])

    output_filename = f'data_h_{h}_n_{n}_thres_{thres}.pickle'
    with open(output_filename, 'wb') as f:
        pickle.dump([U_list, np.array(gap_noH), U_list_anal, gap_anal], f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate gap_noH and gap_anal for given h, n, and thres values.')
    parser.add_argument('-H', '--h', type=float, required=True, help='The h parameter')
    parser.add_argument('-n', '--n', type=int, required=True, help='The n parameter')
    parser.add_argument('-t', '--thres', type=float, default=1e-5, help='The thres parameter (default: 1e-5)')
    

    args = parser.parse_args()

    main(args.h, args.n, args.thres)
