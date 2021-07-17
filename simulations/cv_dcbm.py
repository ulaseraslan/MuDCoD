import sys
import yaml
import argparse

import numpy as np

sys.path.append("../")
from utils.dcbm import simulate_ms_dynamic_dcbm
from MuSPCES.MuSPCES_cv import muspces_cv
from PisCES.PisCES_cv import pisces_cv
from simul_utils import (
    add_common_arguments,
    get_simul_name,
    get_case_path,
    get_config_path,
)


alphalist_pisces = [0.01, 0.02, 0.05, 0.08, 0.1]
alphalist_muspces = [0.01, 0.02, 0.05, 0.08]
betalist_muspces = [0.01, 0.02, 0.05]

parser = argparse.ArgumentParser(description="Cross-validation for MuSPCES and PisCES.")
parser = add_common_arguments(parser)
parser.add_argument(
    "--size-coef",
    required=True,
    type=int,
    default=1,
    help="Size coefficient to determine number of vertices, n_cv=n_case*size-coef.",
)
args = parser.parse_args()

simul_case = args.case
Ns = args.num_subject
T = args.time_horizon
r_subject = args.r_subject
r_time = args.r_time
size_coef = args.size_coef

print(
    f"Simulation case: {simul_case}\n",
    f"Number of subjcets: {Ns}\n",
    f"Time horizon: {T}\n",
    f"Time r value: {r_time}\n",
    f"Subject r value: {r_subject}\n",
    f"Size coefficient: {size_coef}\n",
)

simul_name = get_simul_name(simul_case, T, Ns, r_time, r_subject)
case_path = get_case_path(simul_case)
config_path = get_config_path(simul_case, simul_name)

with open(case_path, "r") as file:
    simul_cfg = yaml.load(file, Loader=yaml.SafeLoader)

K = simul_cfg["K"]
n = int(simul_cfg["n"] * size_coef)
p_in = simul_cfg["p_in"]
p_out = simul_cfg["p_out"]
verbose = simul_cfg["verbose"]

if __name__ == "__main__":
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
    simul_cfg["size_coef"] = size_coef

    with open(config_path, "w") as file:
        yaml.dump(simul_cfg, file, default_flow_style=False)
