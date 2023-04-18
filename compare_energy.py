import argparse
import numpy as np
import pandas as pd
import pickle
from MF2 import *


def convert_pd(spin, digits=6):
    return pd.DataFrame(
        np.round(np.transpose(spin, axes=(0, 2, 1, 3)).reshape((2 * 3, 2 * 3)), 6),
        index=[f"{alpha}{idx}" for alpha in "AB" for idx in range(1, 4)],
        columns=[f"{alpha}{idx}" for alpha in "AB" for idx in range(1, 4)],
    )


def main(nu, n=69, thres=1e-5):
    U_diff_list = np.linspace(0, .5, 6)
    h_list = np.array([.3, .4, .5, .6])

    results = {}

    for i, U_diff in enumerate(U_diff_list):
        for j, h in enumerate(h_list):
            print(U_diff,h)
            params_intraS = Params(
                t=[0, -1, 0],
                U=[0.5 - U_diff, U_diff],
                h=h,
                nu=nu,
                n=n,
            )
            gap_list, energy_list, spin_list, ave = params_intraS.iteration(
                print_opt="", thres=thres
            )

            spin_mat = params_intraS._generate_spin_mat(ave)
            dataframes = [convert_pd(spin_mat[k]) for k in range(4)]

            results[(i, j)] = {
                "U_diff": U_diff,
                "h": h,
                "gap": gap_list[-1],
                "energy": energy_list[-1],
                "ave": ave,
                "spin": dataframes,
            }

    output_filename = f"results_nu_{nu[0]}_{nu[1]}_n_{n}_thres_{thres}.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate spin textures and other properties.")
    parser.add_argument(
        "--nu",
        nargs=2,
        type=int,
        help="Nu list of two components (e.g., [12, 6] or [14, 7], etc.)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=69,
        help="n value (default is 69)",
    )
    parser.add_argument(
        "--thres",
        type=float,
        default=1e-5,
        help="Threshold value (default is 1e-5)",
    )

    args = parser.parse_args()
    main(args.nu, args.n, args.thres)
