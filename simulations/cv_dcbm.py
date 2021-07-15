import os
import sys
import yaml
import argparse

import numpy as np

sys.path.append("../")
from utils.dcbm import simulate_ms_dynamic_dcbm
from MuSPCES.MuSPCES_cv import muspces_cv
from PisCES.PisCES_cv import pisces_cv

parser = argparse.ArgumentParser(description="Cross-validation for MuSPCES and PisCES.")
parser.add_argument("--case", "-c", required=True, type=str, help="Simulation case.")
## parser.add_argument(
##     "--num-subject", "-s", required=True, type=int, help="Number of subjects."
## )
## parser.add_argument(
##     "--time-horizon", "-t", required=True, type=int, help="Number of time steps."
## )
## parser.add_argument(
##     "--r-subject",
##     required=True,
##     type=float,
##     help="Probability of a node changing clusters along subject axis.",
## )
## parser.add_argument(
##     "--r-time",
##     required=True,
##     type=str,
##     help="Probability of a node changing clusters along time axis",
## )
## parser.add_argument(
##     "--num-vertices",
##     "-n",
##     required=True,
##     type=int,
##     help="Total number of vertices in the network.",
## )
## parser.add_argument(
##     "--num-cluster",
##     "-K",
##     required=True,
##     type=int,
##     help="Number of clusters for dcbm and K-means.",
## )
## parser.add_argument(
##     "--p-in",
##     required=True,
##     type=float,
##     nargs="2",
##     help="In-cluster density parameters, p_in=(p_in[0], p_in^[1]).",
## )
## parser.add_argument(
##     "--p-out",
##     required=True,
##     type=float,
##     help="Between-cluster denisty parameter p_out.",
## )
## parser.add_argument(
##     "--size-coef",
##     required=True,
##     type=int,
##     default=1
##     help="Size coefficient to determine number of vertices, n_cv=n_case*size-coef.",
## )
args = parser.parse_args()

simul_case = args.case
Ns = 4
T = 4
r_subject = 0.2
r_time = 0.2
size_coef = 0.2
## K = args.num_cluster
## n = args.num_vertices
## p_in =args.p_in
## p_out =args.p_out

print(f"Simulation case: {simul_case}")

alphalist_pisces = [0.01, 0.02, 0.05, 0.08, 0.1]
alphalist_muspces = [0.02, 0.05, 0.08]
betalist_muspces = [0.01, 0.02, 0.05]

if not os.path.exists("./config_templates"):
    raise ValueError("Directory to read configuration templates does not exist.")

if not os.path.exists("./simulation_configs"):
    raise ValueError("Directory to write simulation configurations does not exist.")

config_path = os.path.join("./config_templates/", simul_case + ".yaml")
config_outpath = os.path.join("./simulation_configs/", simul_case + ".yaml")

if __name__ == "__main__":
    with open(config_path, "r") as file:
        simul_cfg = yaml.load(file, Loader=yaml.SafeLoader)

    K = simul_cfg["K"]
    n = simul_cfg["n"] * size_coef
    p_in = simul_cfg["p_in"]
    p_out = simul_cfg["p_out"]
    ## Ns = simul_cfg["Ns"]
    ## T = simul_cfg["T"]
    ## r_subjcet = simul_cfg["r_subject"]
    ## r_time = simul_cfg["r_time"]
    verbose = simul_cfg["verbose"]

    Adj_series, _ = simulate_ms_dynamic_dcbm(
        Ns, T, K, n, r_subject, r_time, p_in, p_out
    )

    modu_pis = np.zeros(len(alphalist_pisces))
    logllh_pis = np.zeros(len(alphalist_pisces))

    for sbj in range(Ns):
        modu_sbj_pis, logllh_sbj_pis = pisces_cv(
            Adj_series[sbj, :, :, :].astype(float),
            degree_correction=True,
            alphalist=alphalist_pisces,
            verbose=verbose,
        )
        modu_pis = modu_pis + modu_sbj_pis
        logllh_pis = logllh_pis + logllh_sbj_pis

    pis_argmax = np.unravel_index(np.argmax(logllh_pis, axis=None), logllh_pis.shape)
    simul_cfg["alpha_pisces"] = float(alphalist_pisces[pis_argmax[0]])

    modu_musp, logllh_musp = muspces_cv(
        Adj_series.astype(float),
        degree_correction=True,
        alphalist=alphalist_muspces,
        betalist=betalist_muspces,
        verbose=verbose,
    )

    musp_argmax = np.unravel_index(np.argmax(logllh_musp, axis=None), logllh_musp.shape)
    simul_cfg["alpha_muspces"] = float(alphalist_muspces[musp_argmax[0]])
    simul_cfg["beta_muspces"] = float(betalist_muspces[musp_argmax[1]])

    with open(config_outpath, "w") as file:
        yaml.dump(simul_cfg, file, default_flow_style=False)
