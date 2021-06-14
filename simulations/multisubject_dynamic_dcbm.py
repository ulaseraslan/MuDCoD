import os
import sys
import numpy as np
import yaml

from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append("../")
from utils.dcbm import simulate_dynamic_dcbm
from MuSPCES.MuSPCES import muspces
from PisCES.PisCES import pisces
from utils.static import static_clustering


simulation_name = sys.argv[1]
sim_id = sys.argv[2]

if not os.path.exists("./simulation_configs"):
    raise ValueError("Directory read simulation configurations does not exist.")

config_path = os.path.join("./simulation_configs/", simulation_name + ".yaml")

if __name__ == "__main__":
    with open(config_path, "r") as file:
        sim_cfg = yaml.load(file, Loader=yaml.SafeLoader)

    T = sim_cfg["T"]
    Ns = sim_cfg["Ns"]
    K = sim_cfg["K"]
    n = sim_cfg["n"]
    p_in = sim_cfg["p_in"]
    p_out = sim_cfg["p_out"]
    r = sim_cfg["r"]

    alpha_pisces = sim_cfg["alpha_pisces"] * np.ones((T, 2))
    alpha_muspces = sim_cfg["alpha_muspces"] * np.ones((T, 2))
    beta_muspces = sim_cfg["beta_muspces"] * np.ones(T)

    verbose = sim_cfg["verbose"]

    ari_musp = np.empty((Ns, T))
    ari_pis = np.empty((Ns, T))
    ari_static = np.empty((Ns, T))

    Adj_series, z_series = simulate_dynamic_dcbm(T, K, n, r, p_in, p_out, Ns)

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
        z_static[sbj, :, :] = static_clustering(
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

    sim_outdir = os.path.join("./results", simulation_name, "")

    if not os.path.exists(sim_outdir):
        os.mkdir(sim_outdir)

    musp_path = os.path.join(sim_outdir, "muspces" + str(sim_id) + ".csv")
    pis_path = os.path.join(sim_outdir, "pisces" + str(sim_id) + ".csv")
    static_path = os.path.join(sim_outdir, "static" + str(sim_id) + ".csv")

    np.savetxt(musp_path, ari_musp, delimiter=",")
    np.savetxt(pis_path, ari_pis, delimiter=",")
    np.savetxt(static_path, ari_static, delimiter=",")
