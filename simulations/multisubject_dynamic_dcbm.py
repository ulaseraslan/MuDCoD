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
from simul_utils import add_common_arguments, get_simul_name, get_config_path

parser = argparse.ArgumentParser(
    description="Multi-subject Dynamic Degree-corrected Block Model simulation."
)
parser = add_common_arguments(parser)
parser.add_argument("--identity", required=True, type=str, help="Simulation identity.")
args = parser.parse_args()

simul_case = args.case
Ns = args.num_subject
T = args.time_horizon
r_subject = args.r_subject
r_time = args.r_time
simul_identity = args.identity

print(
    f"Simulation case: {simul_case}\n",
    f"Number of subjcets: {Ns}\n",
    f"Time horizon: {T}\n",
    f"Time r value: {r_time}\n",
    f"Subject r value: {r_subject}\n",
    f"Simulation identity number: {simul_identity}\n",
)

simul_name = get_simul_name(simul_case, T, Ns, r_time, r_subject)
config_path = get_config_path(simul_case, simul_name)

with open(config_path, "r") as file:
    simul_cfg = yaml.load(file, Loader=yaml.SafeLoader)

K = simul_cfg["K"]
n = simul_cfg["n"]
p_in = simul_cfg["p_in"]
p_out = simul_cfg["p_out"]
verbose = simul_cfg["verbose"]

if __name__ == "__main__":
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

    simul_outdir = os.path.join("./results", simul_name, "")

    if not os.path.exists(simul_outdir):
        os.mkdir(simul_outdir)

    musp_path = os.path.join(simul_outdir, "muspces_" + str(simul_identity) + ".csv")
    pis_path = os.path.join(simul_outdir, "pisces_" + str(simul_identity) + ".csv")
    static_path = os.path.join(simul_outdir, "static_" + str(simul_identity) + ".csv")

    np.savetxt(musp_path, ari_musp, delimiter=",")
    np.savetxt(pis_path, ari_pis, delimiter=",")
    np.savetxt(static_path, ari_static, delimiter=",")
