import os
import sys
import yaml
import argparse

import numpy as np

from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append("../")
from utils.dcbm import simulate_ms_dynamic_dcbm
from MuSPCES.MuSPCES import muspces
from PisCES.PisCES import pisces
from utils.static import static_spectral_clustering

parser = argparse.ArgumentParser(
    description="Multi-subject Dynamic Degree-corrected Block Model simulation."
)
parser.add_argument(
    "--case", "-c", required=True, type=str, help="Simulation case case."
)
parser.add_argument(
    "--num-subject", "-s", required=True, type=int, help="Number of subjects."
)
parser.add_argument(
    "--time-horizon", "-t", required=True, type=int, help="Number of time steps."
)
parser.add_argument(
    "--r-subject",
    required=True,
    type=float,
    help="Probability of a node changing clusters along subject axis.",
)
parser.add_argument(
    "--r-time",
    required=True,
    type=str,
    help="Probability of a node changing clusters along time axis",
)
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
parser.add_argument("--identity", required=True, type=str, help="Simulation identity.")
args = parser.parse_args()

simul_case = sys.argv[1]
Ns = args.num_subject
T = args.time_horizon
r_subject = args.r_subject
r_time = args.r_time
simul_identity = sys.argv[2]

print(
    f"Simulation case: {simul_case}",
    f"Number of subjcets: {Ns}",
    f"Time horizon: {T}",
    f"Time r value: {r_time}",
    f"Subject r value: {r_subject}",
    f"Simulation identity number: {simul_identity}",
)

if not os.path.exists("./simulation_configs"):
    raise ValueError("Directory to read simulation configurations does not exist.")
else:
    config_path = os.path.join("./simulation_configs/", simul_case + ".yaml")

if __name__ == "__main__":
    with open(config_path, "r") as file:
        simul_cfg = yaml.load(file, Loader=yaml.SafeLoader)

    K = simul_cfg["K"]
    n = simul_cfg["n"]
    p_in = simul_cfg["p_in"]
    p_out = simul_cfg["p_out"]
    ## Ns = simul_cfg["Ns"]
    ## T = simul_cfg["T"]
    ## r_subjcet = simul_cfg["r_subject"]
    ## r_time = simul_cfg["r_time"]
    verbose = simul_cfg["verbose"]

    alpha_pisces = simul_cfg["alpha_pisces"] * np.ones((T, 2))
    alpha_muspces = simul_cfg["alpha_muspces"] * np.ones((T, 2))
    beta_muspces = simul_cfg["beta_muspces"] * np.ones(T)

    ari_musp = np.empty((Ns, T))
    ari_pis = np.empty((Ns, T))
    ari_static = np.empty((Ns, T))

    Adj_series, z_series = simulate_ms_dynamic_dcbm(
        Ns, T, K, n, r_subject, r_time, p_in, p_out
    )

    z_pis = np.empty_like(z_series)
    z_musp = np.empty_like(z_series)
    z_static = np.empty_like(z_series)

    z_musp = muspces(
        Adj_series.astype(float),
        degree_correction=True,
        alpha=alpha_muspces,
        beta=beta_muspces,
        verbose=False,
    )

    for sbj in range(Ns):
        z_pis[sbj, :, :] = pisces(
            Adj_series[sbj, :, :, :].astype(float),
            degree_correction=True,
            alpha=alpha_pisces,
            verbose=False,
        )

    for sbj in range(Ns):
        z_static[sbj, :, :] = static_spectral_clustering(
            Adj_series[sbj, :, :, :].astype(float),
            degree_correction=True,
            verbose=False,
        )

    for sbj in range(Ns):
        for t in range(T):
            ari_pis[sbj, t] = adjusted_rand_score(z_series[sbj, :, t], z_pis[sbj, :, t])
            ari_musp[sbj, t] = adjusted_rand_score(
                z_series[sbj, :, t], z_musp[sbj, :, t]
            )
            ari_static[sbj, t] = adjusted_rand_score(
                z_series[sbj, :, t], z_static[sbj, :, t]
            )

    ari_pis_avg = np.mean(ari_pis)
    ari_musp_avg = np.mean(ari_musp)
    ari_static_avg = np.mean(ari_static)

    if verbose:
        print(f"MuSPCES : {ari_musp_avg}")
        print(f"PisCES  : {ari_pis_avg}")
        print(f"Static : {ari_static_avg}")

    simul_name = simul_case + "_".join(
        [
            "T"
            + str(T)
            + "Ns"
            + str(Ns)
            + "rt"
            + str(r_time)[2:]
            + "rs"
            + str(r_subject)[2:]
        ]
    )
    simul_outdir = os.path.join("./results", simul_name, "")

    if not os.path.exists(simul_outdir):
        os.mkdir(simul_outdir)

    musp_path = os.path.join(simul_outdir, "muspces" + str(simul_identity) + ".csv")
    pis_path = os.path.join(simul_outdir, "pisces" + str(simul_identity) + ".csv")
    static_path = os.path.join(simul_outdir, "static" + str(simul_identity) + ".csv")

    np.savetxt(musp_path, ari_musp, delimiter=",")
    np.savetxt(pis_path, ari_pis, delimiter=",")
    np.savetxt(static_path, ari_static, delimiter=",")
