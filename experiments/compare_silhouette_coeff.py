import argparse
import numpy as np

from sklearn.metrics import silhouette_score

import expt_utils

parser = argparse.ArgumentParser(
    description="Sillhouette score comparison on cell-type specific."
)
parser.add_argument("--verbose", default=False, type=bool)
parser.add_argument(
    "--cell-type",
    required=True,
    type=str,
    help="Cell type that will be used.",
)
parser.add_argument(
    "-percentile",
    required=False,
    type=int,
    default=95,
    help="Percentile used to determine co-expression threshold.",
)

args = parser.parse_args()
cell_type = args.cell_type
percentile = args.percentile
verbose = args.verbose

muspces, pisces, static = expt_utils.get_community_detection_methods(verbose)

data_path = expt_utils.get_data_path(cell_type, percentile)
msdyn_nw_details = expt_utils.get_msdyn_nw_details(data_path)
adj_nw_details = expt_utils.get_adjacency_details(data_path)
msdyn_nw = expt_utils.get_msdyn_nw(data_path)

num_sbj, th, n, _ = msdyn_nw.shape

z_static = np.empty((num_sbj, th, n))
z_pisces = np.empty_like(z_static)
z_muspces = np.empty_like(z_static)

alpha_pisces = 0.1 * np.ones((th, 2))
alpha_muspces = 0.1 * np.ones((th, 2))
beta_muspces = 0.05 * np.ones(num_sbj)

static_sill_scores = []
pisces_sill_scores = []
muspces_sill_scores = []

z_muspces = muspces.fit_predict(
    msdyn_nw[:, :, :, :],
    alpha=alpha_muspces,
    beta=beta_muspces,
    k_max=(n // 10),
)

for sbj in range(num_sbj):
    z_pisces[sbj, :, :] = pisces.fit_predict(
        msdyn_nw[sbj, :, :, :], alpha=alpha_pisces, k_max=n // 10
    )
    for t in range(th):
        z_static[sbj, t, :] = static.fit_predict(msdyn_nw[sbj, t, :, :], k_max=n // 10)
        static_sill_scores.append(
            silhouette_score(
                static.representations[:, : static.model_order_k],
                z_static[sbj, t, :],
            )
        )
        pisces_sill_scores.append(
            silhouette_score(
                pisces.representations[t, :, : pisces.model_order_k[t]],
                z_pisces[sbj, t, :],
            )
        )
        muspces_sill_scores.append(
            silhouette_score(
                muspces.representations[sbj, t, :, : muspces.model_order_k[sbj, t]],
                z_muspces[sbj, t, :],
            )
        )

expt_utils.log(f"mean(silhouette_score)(static):{np.mean(static_sill_scores)}")
expt_utils.log(f"mean(silhouette_score)(pisces):{np.mean(pisces_sill_scores)}")
expt_utils.log(f"mean(silhouette_score)(muspces):{np.mean(muspces_sill_scores)}")
