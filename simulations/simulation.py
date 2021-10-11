import yaml
import logging
import argparse
import numpy as np

from pathlib import Path
from copy import deepcopy
from sklearn.metrics.cluster import adjusted_rand_score
import sys

sys.path.append("../")
from mudcod.dcbm import MuSDynamicDCBM  # noqa: E402
from mudcod.muspces import MuSPCES  # noqa: E402
from mudcod.pisces import PisCES  # noqa: E402
from mudcod.static import Static  # noqa: E402
from mudcod.utils import sutils  # noqa: E402

logging.captureWarnings(True)
MAIN_DIR = Path(__file__).absolute().parent.parent
SIMULATION_DIR = MAIN_DIR / "simulations"
RESULTS_DIR = MAIN_DIR / "results"

N_ITER_CV = 30
N_ITER_CD = 40
ALPHA_VALUES = [0.01, 0.025, 0.05, 0.075, 0.1]
BETA_VALUES = [0.01, 0.025, 0.05, 0.075, 0.1]

parser = argparse.ArgumentParser(
    description="Multi-subject Dynamic Degree-corrected Block Model simulation."
)
parser.add_argument(
    "--results-dir",
    type=str,
    default="",
    help="Path to save the results of simulations.",
)
parser.add_argument("--verbose", default=False, type=bool)
parser.add_argument(
    "--class-dcbm",
    required=True,
    type=str,
    help="DCBM class that will be used.",
)
parser.add_argument(
    "--scenario-msd",
    required=True,
    type=int,
    help="MultiSubject Dynamic DCBM scenario that will be used.",
)
parser.add_argument(
    "--time-horizon", "-t", required=True, type=int, help="Number of time steps."
)
parser.add_argument(
    "--r-time",
    required=True,
    type=float,
    help="Probability of a node changing cluster along the time axis.",
)
parser.add_argument(
    "--num-subjects", "-s", required=True, type=int, help="Number of subjects."
)
parser.add_argument(
    "--r-subject",
    required=True,
    type=float,
    help="Probability of a node changing cluster along the subject axis.",
)
parser.add_argument(
    "--n-jobs",
    required=False,
    type=int,
    default=-2,
    help="Number of concurrently running 'joblib' jobs.",
)

subparsers = parser.add_subparsers(dest="task")
muspces_cv_parser = subparsers.add_parser("cv-muspces")
muspces_cv_parser.add_argument("--alpha", required=True, type=float)
muspces_cv_parser.add_argument("--beta", required=True, type=float)
pisces_cv_parser = subparsers.add_parser("cv-pisces")
pisces_cv_parser.add_argument("--alpha", required=True, type=float)

community_detection_parser = subparsers.add_parser("community-detection")
community_detection_parser.add_argument(
    "--id-number", required=True, type=int, help="Simulation identity number."
)
community_detection_parser.add_argument(
    "--run-cv", dest="run_cv", action="store_true", required=False
)
community_detection_parser.set_defaults(run_cv=False)
community_detection_parser.add_argument(
    "--obj-key",
    type=str,
    default="loglikelihood",
    choices=["loglikelihood", "modularity"],
    help="Objective function to perform cross-validation.",
)

args = parser.parse_args()


def read_class_dcbm(class_dcbm):
    class_dir = SIMULATION_DIR / "classes-DCBM"
    class_path = class_dir / (class_dcbm + ".yaml")
    with open(class_path, "r") as file:
        class_info = yaml.load(file, Loader=yaml.SafeLoader)
    return class_info


def read_cfg(simulation_name):
    cfg_dir = SIMULATION_DIR / "configurations"
    cfg_path = cfg_dir / (simulation_name + ".yaml")
    with open(cfg_path, "r") as file:
        cfg = yaml.load(file, Loader=yaml.SafeLoader)
    return cfg


class SimulationMSDDCBM:
    def __init__(
        self,
        msddcbm_kwargs,
        class_dcbm,
        scenario_msd,
        results_dir=None,
        verbose=False,
        n_jobs=1,
    ):
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.scenario_msd = scenario_msd
        self.msddcbm_kwargs = msddcbm_kwargs
        self.model = MuSDynamicDCBM(**msddcbm_kwargs)
        self.adj_ms_series, self.z_ms_series = self.model.simulate_ms_dynamic_dcbm(
            scenario=self.scenario_msd
        )
        self.class_dcbm = class_dcbm
        self.simulation_name = self._get_simulation_name()

        self.results_dir = Path(results_dir)
        self.simulation_result_path = (
            self.results_dir / "simulation_results" / self.simulation_name
        )
        sutils.ensure_file_dir(self.simulation_result_path)

    def make_cfg(self, obj_key="loglikelihood", run_cv=False):
        cfg_tmp = {}
        cfg_tmp["scenario_msd"] = self.scenario_msd
        cfg_tmp["time_horizon"] = self.model.time_horizon
        cfg_tmp["r_time"] = self.model.r_time
        cfg_tmp["num_subjects"] = self.model.num_subjects
        cfg_tmp["r_subject"] = self.model.r_subject
        cfg = {**deepcopy(read_class_dcbm(self.class_dcbm)), **cfg_tmp}

        cfg_dir = SIMULATION_DIR / "configurations"
        simulation_name = SimulationMSDDCBM.get_simulation_name(
            self.class_dcbm, **cfg_tmp
        )
        cfg_path = cfg_dir / (simulation_name + ".yaml")

        if run_cv:
            if obj_key == "loglikelihood":
                obj_idx = 0
            elif obj_key == "modularity":
                obj_idx = 1
            else:
                assert False
            pisces_obj_max = -np.inf
            muspces_obj_max = -np.inf
            for alpha in ALPHA_VALUES:
                pisces_obj_vals = self.cv_pisces(alpha, save=True)
                if pisces_obj_vals[obj_idx] > pisces_obj_max:
                    cfg["alpha_pisces"] = alpha
                for beta in BETA_VALUES:
                    muspces_obj_vals = self.cv_muspces(alpha, beta, save=True)
                    if muspces_obj_vals[obj_idx] > muspces_obj_max:
                        cfg["alpha_muspces"] = alpha
                        cfg["beta_muspces"] = beta
        else:
            simulation_result_dir = self.results_dir / "simulation_results"
            cv_result_dir = simulation_result_dir / simulation_name / "cross_validation"

            if not cv_result_dir.exists():
                raise FileNotFoundError(
                    "Corresponding cross-validation result directory can not be found."
                )
            pisces_obj_max = -np.inf
            muspces_obj_max = -np.inf
            for cv_result_path in sorted(cv_result_dir.glob("*.csv")):
                cv_name = cv_result_path.stem
                method = cv_name.split("_")[0].lower()
                cv_result = {
                    key: list(map(float, values))
                    for key, values in sutils.read_csv_to_dict(cv_result_path).items()
                }

                ## obj = np.mean(cv_result[obj_key])
                obj = cv_result[obj_key][-1]
                if method == "muspces":
                    assert len(set(cv_result["alpha"])) <= 1
                    assert len(set(cv_result["beta"])) <= 1
                    if muspces_obj_max < obj:
                        cfg["alpha_muspces"] = cv_result["alpha"][0]
                        cfg["beta_muspces"] = cv_result["beta"][0]
                        muspces_obj_max = obj
                elif method == "pisces":
                    assert len(set(cv_result["alpha"])) <= 1
                    if pisces_obj_max < obj:
                        cfg["alpha_pisces"] = cv_result["alpha"][0]
                        pisces_obj_max = obj
                else:
                    raise ValueError(
                        f"Unkown method {method} is encountered in the cross-validation "
                        f"result directory {cv_result_dir}."
                    )

        sutils.ensure_file_dir(cfg_path)
        with open(cfg_path, "w") as file:
            yaml.dump(cfg, file, default_flow_style=False)
        sutils.log(f"Configuration file created in {cfg_path}.")
        return cfg

    @staticmethod
    def get_simulation_name(
        class_dcbm, scenario_msd, time_horizon, r_time, num_subjects, r_subject
    ):
        simulation_name = "_".join(
            [
                class_dcbm,
                "scenario" + str(scenario_msd),
                "th" + str(time_horizon),
                "rt" + str(r_time)[2:],
                "ns" + str(num_subjects),
                "rs" + str(r_subject)[2:],
            ]
        )
        return simulation_name

    def _get_simulation_name(self):
        return self.get_simulation_name(
            self.class_dcbm,
            self.scenario_msd,
            self.model.time_horizon,
            self.model.r_time,
            self.model.num_subjects,
            self.model.r_subject,
        )

    @sutils.timeit
    def cv_muspces(self, alpha, beta, save=True):
        adj_ms_series = self.adj_ms_series
        ns = self.model.num_subjects
        th = self.model.time_horizon

        muspces = MuSPCES(verbose=self.verbose)
        muspces.fit(deepcopy(adj_ms_series), n_iter=0, degree_correction=False)

        modu_muspces, logllh_muspces = muspces.cross_validation(
            alpha=alpha * np.ones((th, 2)),
            beta=beta * np.ones(ns),
            n_jobs=self.n_jobs,
            n_iter=N_ITER_CV,
        )
        if save:
            name = "_".join(
                [
                    "muspces",
                    "alpha" + str(alpha)[2:],
                    "beta" + str(beta)[2:],
                ]
            )
            header = ["alpha", "beta", "modularity", "loglikelihood"]
            values = [
                alpha,
                beta,
                modu_muspces,
                logllh_muspces,
            ]
            self.save_cross_validation_result(name, header, values)

        return modu_muspces, logllh_muspces

    @sutils.timeit
    def cv_pisces(self, alpha, save=True):
        adj_ms_series = self.adj_ms_series
        ns = self.model.num_subjects
        th = self.model.time_horizon

        pisces = PisCES(verbose=self.verbose)

        modu_pisces, logllh_pisces = 0, 0

        for sbj in range(ns):
            pisces.fit(adj_ms_series[sbj, :, :, :], n_iter=0, degree_correction=False)
            modu_sbj_pisces, logllh_sbj_pisces = pisces.cross_validation(
                alpha=alpha * np.ones((th, 2)), n_jobs=self.n_jobs, n_iter=N_ITER_CV
            )
            modu_pisces = modu_pisces + modu_sbj_pisces
            logllh_pisces = logllh_pisces + logllh_sbj_pisces

        if save:
            name = "_".join(
                [
                    "pisces",
                    "alpha" + str(alpha)[2:],
                ]
            )
            header = ["alpha", "modularity", "loglikelihood"]
            values = [alpha, modu_pisces, logllh_pisces]
            self.save_cross_validation_result(name, header, values)

        return modu_pisces, logllh_pisces

    @sutils.timeit
    def community_detection(
        self,
        id_number,
        alpha_pisces=None,
        alpha_muspces=None,
        beta_muspces=None,
        save=True,
    ):
        th = self.model.time_horizon
        ns = self.model.num_subjects
        alpha_pisces = alpha_pisces * np.ones((th, 2))
        alpha_muspces = alpha_muspces * np.ones((th, 2))
        beta_muspces = beta_muspces * np.ones(ns)

        ari_muspces = np.empty((ns, th))
        ari_pisces = np.empty((ns, th))
        ari_static = np.empty((ns, th))

        z_ms_series = self.z_ms_series
        adj_ms_series = self.adj_ms_series
        ns = adj_ms_series.shape[0]
        th = adj_ms_series.shape[1]

        z_pisces = np.empty_like(z_ms_series)
        z_muspces = np.empty_like(z_ms_series)
        z_static = np.empty_like(z_ms_series)

        muspces = MuSPCES(verbose=self.verbose)
        pisces = PisCES(verbose=self.verbose)
        static = Static(verbose=self.verbose)

        z_muspces = muspces.fit_predict(
            deepcopy(adj_ms_series),
            alpha=alpha_muspces,
            beta=beta_muspces,
            k_max=self.model.n // 10,
            n_iter=N_ITER_CD,
        )

        if self.n_jobs > 1:
            from joblib import Parallel, delayed

            with Parallel(n_jobs=self.n_jobs) as parallel:
                temp_pisces = parallel(
                    delayed(pisces.fit_predict)(
                        deepcopy(adj_ms_series[sbj, :, :, :]),
                        alpha=alpha_pisces,
                        k_max=self.model.n // 10,
                        n_iter=N_ITER_CD,
                    )
                    for sbj in range(ns)
                )
                z_pisces = np.array(temp_pisces)

            with Parallel(n_jobs=self.n_jobs) as parallel:
                temp_static = parallel(
                    delayed(static.fit_predict)(
                        deepcopy(adj_ms_series[sbj, t, :, :]), k_max=self.model.n // 10
                    )
                    for sbj in range(ns)
                    for t in range(th)
                )
                z_static = np.array(temp_static).reshape((ns, th, -1))

        else:
            for sbj in range(ns):
                z_pisces[sbj, :, :] = pisces.fit_predict(
                    adj_ms_series[sbj, :, :, :],
                    alpha=alpha_pisces,
                    k_max=self.model.n // 10,
                    n_iter=N_ITER_CD,
                )

            for sbj in range(ns):
                for t in range(th):
                    z_static[sbj, t, :] = static.fit_predict(
                        adj_ms_series[sbj, t, :, :], k_max=self.model.n // 10
                    )

        for t in range(th):
            for sbj in range(ns):
                ari_muspces[sbj, t] = adjusted_rand_score(
                    z_ms_series[sbj, t, :], z_muspces[sbj, t, :]
                )
                ari_pisces[sbj, t] = adjusted_rand_score(
                    z_ms_series[sbj, t, :], z_pisces[sbj, t, :]
                )
                ari_static[sbj, t] = adjusted_rand_score(
                    z_ms_series[sbj, t, :], z_static[sbj, t, :]
                )

        ari_muspces_avg = np.mean(ari_muspces)
        ari_pisces_avg = np.mean(ari_pisces)
        ari_static_avg = np.mean(ari_static)

        if self.verbose:
            sutils.log(
                f"ARI(MuSPCES): {ari_muspces_avg}, "
                f"ARI(PisCES): {ari_pisces_avg}, "
                f"ARI(Static): {ari_static_avg}"
            )
        if save:
            self.save_community_detection_result("muspces", id_number, ari_muspces)
            self.save_community_detection_result("pisces", id_number, ari_pisces)
            self.save_community_detection_result("static", id_number, ari_static)

        return ari_muspces, ari_pisces, ari_static

    def save_community_detection_result(self, name, id_number, result):
        savedir = self.simulation_result_path / "community_detection"
        path = savedir / "_".join([name, str(id_number) + ".csv"])
        sutils.ensure_file_dir(path)
        np.savetxt(path, result, delimiter=",", fmt="%s")

    def save_cross_validation_result(self, name, header, values):
        savedir = self.simulation_result_path / "cross_validation"
        path = savedir / (name + ".csv")
        sutils.write_to_csv(path, header, values)


if __name__ == "__main__":
    results_dir = args.results_dir
    task = args.task
    class_dcbm = args.class_dcbm
    time_horizon = args.time_horizon
    r_time = args.r_time
    num_subjects = args.num_subjects
    r_subject = args.r_subject
    scenario_msd = args.scenario_msd
    n_jobs = args.n_jobs
    verbose = args.verbose

    if results_dir == "":
        results_dir = Path(RESULTS_DIR)

    sutils.log(f"Results will be saved to:{results_dir}")
    sutils.log(
        f"DCBM-class-name:{class_dcbm}, task:{task}, "
        f"time-horizon:{time_horizon}, r-time:{r_time}, "
        f"num-subjects:{num_subjects}, r-subject:{r_subject}, "
        f"scenario-MSD:{scenario_msd}, n-jobs:{n_jobs}, verbose:{verbose}"
    )

    class_info = read_class_dcbm(class_dcbm)
    msddcbm_kwargs = deepcopy(class_info)
    msddcbm_kwargs["time_horizon"] = time_horizon
    msddcbm_kwargs["r_time"] = r_time
    msddcbm_kwargs["num_subjects"] = num_subjects
    msddcbm_kwargs["r_subject"] = r_subject

    Simul = SimulationMSDDCBM(
        msddcbm_kwargs,
        class_dcbm,
        scenario_msd,
        results_dir=results_dir,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    Simul.parser_args = args

    if task == "cv-pisces":
        alpha = args.alpha
        sutils.log(f"alpha:{alpha}")
        Simul.cv_pisces(alpha, save=True)

    elif task == "cv-muspces":
        alpha = args.alpha
        beta = args.beta
        sutils.log(f"alpha:{alpha}, beta:{beta}")
        Simul.cv_muspces(alpha, beta, save=True)

    elif task == "community-detection":
        obj_key = args.obj_key
        id_number = args.id_number
        run_cv = args.run_cv
        cfg = Simul.make_cfg(obj_key=obj_key, run_cv=run_cv)
        sutils.log(f"For PisCES; alpha:{cfg['alpha_pisces']}")
        sutils.log(
            f"For MuSPCES; alpha:{cfg['alpha_muspces']}, "
            f" beta:{cfg['beta_muspces']}"
        )
        sutils.log(f"objective-key:{obj_key}, identity-number:{id_number}")
        Simul.community_detection(
            id_number,
            alpha_pisces=cfg["alpha_pisces"],
            alpha_muspces=cfg["alpha_muspces"],
            beta_muspces=cfg["beta_muspces"],
            save=True,
        )
    else:
        assert False
