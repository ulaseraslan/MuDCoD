import os
import glob
import pandas as pd
import numpy as np


from collections import defaultdict


def read_results_to_df(input_path="../simulations/results/"):
    """
    Read mean(ARI) of each MuSPCES, PisCES, static to pandas DataFrame.
    """
    result_dirs = [f.path for f in os.scandir(input_path) if f.is_dir()]
    result_dict = defaultdict(list)

    for i, path in enumerate(sorted(result_dirs)):
        simul_name = os.path.basename(os.path.normpath(path))
        if not simul_name.startswith("."):
            simul_info = simul_name.split("_")
            case, Th, rs, rt  = simul_info[0], simul_info[1], simul_info[2], simul_info[3],
            # TODO: Fix this!
            case, Ns = case[:-3], case[-3:]
            print(case, Ns, Th, rs, rt)

            muspces_results_path = sorted(glob.glob(path + "/muspces*.csv"))
            muspces = np.empty(len(muspces_results_path))
            for i, mpath in enumerate(sorted(muspces_results_path)):
                temp = np.genfromtxt(mpath, delimiter=",")
                muspces[i] = np.mean(temp)
            result_dict["row"].append((case, rs, rt, Ns, Th, "muspces"))
            result_dict["val"].append(np.mean(muspces))

            pisces_results_path = sorted(glob.glob(path + "/pisces*.csv"))
            pisces = np.empty(len(pisces_results_path))
            for i, ppath in enumerate(sorted(pisces_results_path)):
                temp = np.genfromtxt(ppath, delimiter=",")
                pisces[i] = np.mean(temp)
            result_dict["row"].append((case, rs, rt, Ns, Th, "pisces"))
            result_dict["val"].append(np.mean(pisces))


            static_results_path = sorted(glob.glob(path + "/static*.csv"))
            static = np.empty(len(static_results_path))
            for i, spath in enumerate(sorted(static_results_path)):
                temp = np.genfromtxt(spath, delimiter=",")
                static[i] = np.mean(temp)
            result_dict["row"].append((case, rs, rt, Ns, Th, "static"))
            result_dict["val"].append(np.mean(static))

    index_row = pd.MultiIndex.from_tuples(result_dict["row"], names=["case", "r_sbj", "r_time", "num-sbj", "time-horizon", "method"])

    resultsDf = pd.DataFrame(result_dict["val"], index=index_row, columns=["mean(ARI)"])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(resultsDf.loc["easy200K5", "rs001", "rt001", :, :, :])
    return resultsDf

if __name__=="__main__":
    read_results_to_df()
