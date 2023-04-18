import argparse
import numpy as np
import pandas as pd
from MF import *
import pickle
import time

def convert_pd(spin, Nq,digits=6):
    return pd.DataFrame(np.round(np.transpose(spin, axes=(0, 2, 1, 3)).reshape((2 * Nq, 2 * Nq)), 2* Nq),
                        index=[f'{alpha}{idx}' for alpha in 'AB' for idx in range(1, Nq+1)],
                        columns=[f'{alpha}{idx}' for alpha in 'AB' for idx in range(1, Nq+1)])

def main(args):
    t1_list = np.linspace(*args.t1[:2], int(args.t1[-1]))
    t2_list = np.linspace(*args.t2[:2], int(args.t2[-1]))
    U0_list = np.linspace(*args.U0[:2], int(args.U0[-1]))
    U1_list = np.linspace(*args.U1[:2], int(args.U1[-1]))
    results = {}

    for t1_idx, t1 in enumerate(t1_list):
        for t2_idx, t2 in enumerate(t2_list):
            for U0_idx, U0 in enumerate(U0_list):
                for U1_idx, U1 in enumerate(U1_list):
                    print(f"Current iteration: t1={t1}, t2={t2}, U0={U0}, U1={U1}",flush=True)
                    params = Params(t=[0, t1, t2], U=[U0, U1], h=args.h, nu=args.nu, n=args.n,hartree=args.hartree,fock=args.fock)
                    gap_list, energy_list, spin_list, ave = params.iteration(print_opt='m', thres=args.thres)
                    spin_mat = params._generate_spin_mat(ave)
                    spin = []
                    for i in range(4):
                        spin.append(convert_pd(spin_mat[i],Nq=params.q.shape[0]))
                    results[(t1_idx, t2_idx, U0_idx, U1_idx)] = {
                        't1': t1,
                        't2': t2,
                        'U0': U0,
                        'U1': U1,
                        'gap': gap_list[-1],
                        'energy': energy_list[-1],
                        'ave': ave,
                        'spin': spin
                    }

    filename = f"results_nu({args.nu[0]},{args.nu[1]})_t1({args.t1[0]},{args.t1[1]},{int(args.t1[2])})_t2({args.t2[0]},{args.t2[1]},{int(args.t2[2])})_U0({args.U0[0]},{args.U0[1]},{int(args.U0[2])})_U1({args.U1[0]},{args.U1[1]},{int(args.U1[2])})_h{args.h}_thres{args.thres}_n{args.n}{'_noH' if not args.hartree else ''}{'_noF' if not args.fock else ''}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterate parameters and compute spin texture.')
    parser.add_argument('--nu', nargs=2, type=int, help='List of two components for nu.')
    parser.add_argument('-t1', nargs=3, type=float, help='Start, stop, num for t1.')
    parser.add_argument('-t2', nargs=3, type=float, help='Start, stop, num for t2.')
    parser.add_argument('-U0', nargs=3, type=float, help='Start, stop, num for U0.')
    parser.add_argument('-U1', nargs=3, type=float, help='Start, stop, num for U1.')
    parser.add_argument('-H', '--h', type=float, help='Zeeman field h.')
    parser.add_argument('--thres', type=float, default=1e-5, help='Threshold for convergence, default is 1e-5.')
    parser.add_argument('-n', type=int, help='Number of iterations.')
    parser.add_argument('--no-hartree', dest='hartree', action='store_false', default=True, help='Disable hartree (default: enabled)')
    parser.add_argument('--no-fock', dest='fock', action='store_false', default=True, help='Disable fock (default: enabled)')
    args = parser.parse_args()
    st=time.time()
    main(args)
    print(f'Elapsed Time:{time.time()-st}s. Total points: {args.t1[2]*args.t2[2]*args.U0[2]*args.U1[2]}')
