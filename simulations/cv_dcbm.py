import os
import sys
import yaml
import numpy as np

sys.path.append("../")
from utils.dcbm import simulate_dynamic_dcbm
from MuSPCES.MuSPCES_cv import muspces_cv
from PisCES.PisCES_cv import pisces_cv

alphalist_pisces = [0.01, 0.02, 0.05, 0.08, 0.1]
alphalist_muspces = [
    0.02,
    0.05,
    0.08,
]
betalist_muspces = [
    0.01,
    0.02,
    0.05,
]

if len(sys.argv) < 5:
    simulation_name = sys.argv[1]
    T = int(sys.argv[2])
    Ns = int(sys.argv[3])
else:
    raise ValueError("Requires excatly 3 arguments.")

if not os.path.exists("./config_templates"):
    raise ValueError("Directory to read configuration templates does not exist.")

if not os.path.exists("./simulation_configs"):
    raise ValueError("Directory to write simulation configurations does not exist.")

config_path = os.path.join("./config_templates/", simulation_name + ".yaml")
config_outpath = os.path.join(
    "./simulation_configs/",
    simulation_name + "_T" + str(T) + "_Ns" + str(Ns) + ".yaml",
)

if __name__ == "__main__":
    with open(config_path, "r") as file:
        sim_cfg = yaml.load(file, Loader=yaml.SafeLoader)

    K = sim_cfg["K"]
    n = sim_cfg["n"]
    p_in = sim_cfg["p_in"]
    p_out = sim_cfg["p_out"]
    r = sim_cfg["r"]

    verbose = sim_cfg["verbose"]

    Adj_series, _ = simulate_dynamic_dcbm(T, K, n, r, p_in, p_out, Ns)

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
    sim_cfg["alpha_pisces"] = float(alphalist_pisces[pis_argmax[0]])

    modu_musp, logllh_musp = muspces_cv(
        Adj_series.astype(float),
        degree_correction=True,
        alphalist=alphalist_muspces,
        betalist=betalist_muspces,
        verbose=verbose,
    )

    musp_argmax = np.unravel_index(np.argmax(logllh_musp, axis=None), logllh_musp.shape)
    sim_cfg["alpha_muspces"] = float(alphalist_muspces[musp_argmax[0]])
    sim_cfg["beta_muspces"] = float(betalist_muspces[musp_argmax[1]])

    sim_cfg["T"] = T
    sim_cfg["Ns"] = Ns

    with open(config_outpath, "w") as file:
        yaml.dump(sim_cfg, file, default_flow_style=False)
