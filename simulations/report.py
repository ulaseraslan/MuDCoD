import sys
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path
from collections import defaultdict

MAIN_DIR = Path(__file__).absolute().parent.parent
SIMULATION_DIR = MAIN_DIR / "simulations"
RESULT_DIR = MAIN_DIR / "results"
FIGURE_DIR = RESULT_DIR / "simulation_figures"
sys.path.append(str(MAIN_DIR))


from dypoces.utils import sutils  # noqa: E402


sutils.safe_create_dir(FIGURE_DIR)
sns.set_theme(style="whitegrid")


def read_simulation_results_to_df(simul_result_path, multi_index=False):
    # {{{
    """
    Read mean(ARI) of each MuSPCES, PisCES, static to pandas DataFrame.
    """
    simul_result_path = Path(simul_result_path)
    result_dirs = [f for f in simul_result_path.iterdir() if f.is_dir()]
    result_dict = defaultdict(list)
    num_result = len(result_dirs)

    mkint = lambda x: int(x[2:])
    mkfloat = lambda x: float("0." + x[2:])

    for i, path in enumerate(sorted(result_dirs)):
        simul_name = str(path.stem)
        simul_info = simul_name.split("_")
        if not simul_name.startswith(".") and len(simul_info) == 6:
            class_dcbm, case_msd, th, rt, ns, rs = (
                simul_info[0],
                simul_info[1],
                simul_info[2],
                simul_info[3],
                simul_info[4],
                simul_info[5],
            )

            percent = round(100 * i / num_result, 2)
            print(
                f"Procesing:%{percent}", class_dcbm, case_msd, th, rt, ns, rs, end="\r"
            )
            communities_path = path / "communities"
            ns, th = mkint(ns), mkint(th)
            rs, rt = mkfloat(rs), mkfloat(rt)

            muspces_results_path = communities_path.glob("muspces*.csv")
            for i, mpath in enumerate(sorted(muspces_results_path)):
                temp = np.genfromtxt(mpath, delimiter=",")
                result_dict["row"].append(
                    (i, class_dcbm, case_msd, th, rt, ns, rs, "muspces")
                )
                result_dict["val"].append(np.mean(temp))

            pisces_results_path = communities_path.glob("pisces*.csv")
            for i, ppath in enumerate(sorted(pisces_results_path)):
                temp = np.genfromtxt(ppath, delimiter=",")
                result_dict["row"].append(
                    (i, class_dcbm, case_msd, th, rt, ns, rs, "pisces")
                )
                result_dict["val"].append(np.mean(temp))

            static_results_path = communities_path.glob("static*.csv")
            for i, spath in enumerate(sorted(static_results_path)):
                temp = np.genfromtxt(spath, delimiter=",")
                result_dict["row"].append(
                    (i, class_dcbm, case_msd, th, rt, ns, rs, "static")
                )
                result_dict["val"].append(np.mean(temp))

    index_row = pd.MultiIndex.from_tuples(
        result_dict["row"],
        names=[
            "id",
            "class-dcbm",
            "case-msd",
            "time-horizon",
            "r_time",
            "num-subjects",
            "r_subject",
            "method",
        ],
    )
    resultsDf = pd.DataFrame(result_dict["val"], index=index_row, columns=["mean(ARI)"])

    if not multi_index:
        resultsDf.reset_index(inplace=True)

    return resultsDf


# }}}


results_df = read_simulation_results_to_df(RESULT_DIR / "simulation_results")

unique_col_val = {}
for col in results_df.columns:
    unique_col_val[col] = sorted(results_df[col].unique())

x = "time-horizon"
y = "mean(ARI)"
hue = "method"
col = "r_time"
row = "r_subject"

for class_dcbm in unique_col_val["class-dcbm"]:
    for case_msd in unique_col_val["case-msd"]:
        for ns in unique_col_val["num-subjects"]:
            mask_trip = (
                (results_df["class-dcbm"] == class_dcbm)
                & (results_df["case-msd"] == case_msd)
                & (results_df["num-subjects"] == ns)
            )
            data = results_df[mask_trip]
            g = sns.catplot(
                data=data,
                x=x,
                y=y,
                hue=hue,
                col=col,
                row=row,
                palette="hls",
                kind="point",
                capsize=0.2,
                aspect=0.75,
                height=6,
            )
            title = f"dcbm-class: {class_dcbm} "
            title += f"case-msd: {case_msd} "
            title += f"num-subjects: {str(ns)}"
            g.fig.suptitle(title, y=1.05)
            g.despine(left=True)
            g.fig.savefig(
                FIGURE_DIR / f"{class_dcbm}_{case_msd}_ns{ns}.png", bbox_inches="tight"
            )
