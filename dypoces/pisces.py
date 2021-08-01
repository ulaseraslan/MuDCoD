import warnings
import numpy as np

from copy import deepcopy
from sklearn.cluster import KMeans
from numpy.linalg import inv, eigvals
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm

from dypoces.spectral import SpectralClustering
from dypoces.dcbm import DCBM
from dypoces.utils.sutils import timeit, log


warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

_eps = 10 ** (-10)


class PisCES(SpectralClustering):
    def __init__(self, verbose=False):
        super().__init__("pisces", verbose=verbose)

    def fit(self, adj, degree_correction=True):
        """
        Parameters
        ----------
        adj
            adjacency matrices with dimention (th,n,n),
            n is the number of nodes and th is the number of time time steps.
        degree_correction
            Laplacianized, default is 'True'.
        """
        assert type(adj) in [np.ndarray, np.memmap] and adj.ndim == 3

        self.adj = adj.astype(float)
        self.adj_shape = self.adj.shape
        self.n = self.adj_shape[1]  # or adj_shape[2]
        self.time_horizon = self.adj_shape[0]
        self.degree_correction = degree_correction
        self.degree = deepcopy(adj)

        if self.verbose:
            log(f"PisCES-fit::: #nodes:{self.n}, " f"#time:{self.time_horizon}")

        if self.degree_correction:
            for t in range(self.time_horizon):
                adj_t = self.adj[t, :, :]
                dg = np.diag(np.sum(np.abs(adj_t), axis=0) + _eps)
                sqinv_degree = sqrtm(inv(dg))
                self.adj[t, :, :] = sqinv_degree @ adj_t @ sqinv_degree
                self.degree[t, :, :] = dg
        else:
            for t in range(self.time_horizon):
                self.degree[t, :, :] = np.eye(self.n)

    def predict(self, alpha=None, k_max=None, n_iter=30):
        """
        Parameters
        ----------
        alpha
            smoothing tuning parameter, along time axis, default is
            0.05J(th,2).
        k_max
            maximum number of communities, default is n/10.
        n_iter
            number of iteration of pisces, default is 30.

        Returns
        -------
        z_series: community prediction for each time point, with shape (th, n).
        """
        th = self.time_horizon
        n = self.n
        adj = self.adj
        degree = self.degree

        if alpha is None:
            log(f"alpha is not provided, alpha set to 0.05J({th},2) by default.")
            alpha = 0.05 * np.ones((th, 2))
        if k_max is None:
            k_max = n // 10
            log(f"k_max is not provided, default value is floor({n}/10).")

        if th < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral "
                "clustering."
            )

        assert alpha.shape == (th, 2)
        assert k_max > 0

        if self.verbose:
            log(
                f"PisCES-predict::: alpha:{alpha[0,0]}, "
                f"k_max {k_max}, n_iter:{n_iter}"
            )

        k = np.zeros(th).astype(int) + k_max
        obj = np.zeros(n_iter)
        v_emb = np.zeros((th, n, k_max))

        # Initialization of k, v_emb.
        for t in range(th):
            adj_t = adj[t, :, :]
            k[t] = self.choose_k(adj_t, adj_t, degree[t, :, :], k_max)
            _, v_emb[t, :, : k[t]] = eigs(adj_t, k=k[t], which="LM")

        for itr in range(n_iter):
            v_emb_pv = deepcopy(v_emb)
            for t in range(th):
                if t == 0:
                    adj_t = adj[t, :, :]
                    v_emb_ktn = v_emb_pv[t + 1, :, : k[t + 1]]
                    s_t = adj_t + alpha[t, 1] * v_emb_ktn @ v_emb_ktn.T
                    k[t] = self.choose_k(s_t, adj_t, degree[t, :, :], k_max)
                    _, v_emb[t, :, : k[t]] = eigs(adj_t, k=k[t], which="LM")
                    eig_val = eigvals(v_emb[t, :, : k[t]].T @ v_emb_pv[t, :, : k[t]])
                    obj[itr] = obj[itr] + (np.sum(np.abs(eig_val), axis=0))

                elif t == th - 1:
                    adj_t = adj[t, :, :]
                    v_emb_ktp = v_emb_pv[t - 1, :, : k[t - 1]]
                    s_t = adj_t + alpha[t, 0] * v_emb_ktp @ v_emb_ktp.T
                    k[t] = self.choose_k(s_t, adj_t, degree[t, :, :], k_max)
                    _, v_emb[t, :, : k[t]] = eigs(s_t, k=k[t], which="LM")
                    eig_val = eigvals(v_emb[t, :, : k[t]].T @ v_emb_pv[t, :, : k[t]])
                    obj[itr] = obj[itr] + (np.sum(np.abs(eig_val), axis=0))

                else:
                    adj_t = adj[t, :, :]
                    v_emb_ktp = v_emb_pv[t - 1, :, : k[t - 1]]
                    v_emb_ktn = v_emb_pv[t + 1, :, : k[t + 1]]
                    s_t = (
                        adj_t
                        + (alpha[t, 0] * v_emb_ktp @ v_emb_ktp.T)
                        + (alpha[t, 1] * v_emb_ktn @ v_emb_ktn.T)
                    )
                    k[t] = self.choose_k(s_t, adj_t, degree[t, :, :], k_max)
                    _, v_emb[t, :, : k[t]] = eigs(s_t, k=k[t], which="LM")
                    eig_val = eigvals(v_emb[t, :, : k[t]].T @ v_emb_pv[t, :, : k[t]])
                    obj[itr] = obj[itr] + np.sum(np.abs(eig_val), axis=0)

            if self.verbose:
                log(f"Value of objective funciton: {obj[itr]}, at iteration {itr+1}.")

            if itr > 1 and abs(obj[itr] - obj[itr - 1]) < 0.001:
                break

        if obj[itr] - obj[itr - 1] >= 0.001:
            warnings.warn("PisCES does not converge!", RuntimeWarning)
            if self.verbose:
                log(f"PisCES does not not converge for alpha={alpha[0, 0]}.")

        z_series = np.empty((th, n), dtype=int)
        for t in range(th):
            kmeans = KMeans(n_clusters=k[t])
            z_series[t, :] = kmeans.fit_predict(v_emb[t, :, : k[t]])

        return z_series

    def fit_predict(
        self, adj, degree_correction=True, alpha=None, k_max=None, n_iter=30
    ):
        self.fit(adj, degree_correction=degree_correction)
        return self.predict(alpha=alpha, k_max=k_max, n_iter=n_iter)

    def cross_validation(
        self,
        n_splits=5,
        alpha=None,
        k_max=None,
        n_iter=30,
        n_jobs=1,
    ):
        """
        This is a function for cross validation of PisCES method
        Parameters
        ----------
        n_splits
                number of folds in cross validation
        alpha
            smoothing tuning parameter, along time axis, default is
            0.05J(th,2).
        k_max
                maximum number of communities, default is N/10
        n_iter
                number of iteration of pisces, default is 10
        n_jobs
                number of parallel joblib jobs

        Returns
        -------
        modu
                modularity value for alpha
        logllh
                log likelihood value for alpha
        """

        th = self.time_horizon
        n = self.n
        adj = self.adj

        if alpha is None:
            log(f"alpha is not provided, alpha set to 0.05J({th},2) by default.")
            alpha = 0.05 * np.ones((th, 2))
        if k_max is None:
            k_max = n // 10
            log(f"k_max is not provided, default value is floor({n}/10).")

        if th < 2:
            raise ValueError(
                "Time horizon must be at least 2, otherwise use static spectral "
                "clustering."
            )

        assert alpha.shape == (th, 2)
        assert k_max > 0

        idx_n = np.arange(n)
        idx = np.c_[np.repeat(idx_n, idx_n.shape), np.tile(idx_n, idx_n.shape)]
        r = np.random.choice(n ** 2, size=n ** 2)

        pisces_kwargs = {
            "alpha": alpha,
            "k_max": k_max,
            "n_iter": n_iter,
        }

        def compute_for_split(adj, idx_test, n, th, pisces_kwargs={}):
            cvidx = np.empty((th, n, n))
            adj_train = np.zeros((th, n, n))
            adj_train_cv = np.zeros((th, n, n))

            for t in range(th):
                cvidx_t = np.zeros((n, n))
                cvidx_t[idx_test[:, 0], idx_test[:, 1]] = 1
                cvidx_t = np.triu(cvidx_t) + np.triu(cvidx_t).T
                cvidx[t, :, :] = cvidx_t

                adj_t = deepcopy(adj[t, :, :])
                adj_t[idx_test[:, 0], idx_test[:, 1]] = 0
                adj_t = np.triu(adj_t) + np.triu(adj_t).T
                adj_train[t, :, :] = adj_t
                adj_train_cv[t, :, :] = self.eigen_complete(adj_t, cvidx_t, 10, 10)

            z = self.__class__(verbose=self.verbose).fit_predict(
                adj_train_cv[:, :, :],
                degree_correction=self.degree_correction,
                **pisces_kwargs,
            )

            modu_val, logllh_val = 0, 0
            for t in range(th):
                modu_val = modu_val + DCBM.loss(
                    adj[t, :, :],
                    adj_train[t, :, :],
                    z[t, :],
                    cvidx[t, :, :],
                    opt="modu",
                )
                logllh_val = logllh_val + DCBM.loss(
                    adj[t, :, :],
                    adj_train[t, :, :],
                    z[t, :],
                    cvidx[t, :, :],
                    opt="logllh",
                )
            return modu_val, logllh_val

        modu = 0
        logllh = 0

        def split_idx_test(split):
            psplit = n ** 2 // n_splits
            test = r[split * psplit : (split + 1) * psplit]
            idx_test = idx[test, :]
            return idx_test

        if n_jobs > 1:
            from joblib import Parallel, delayed

            with Parallel(n_jobs=n_jobs) as parallel:  ## prefer="processes"
                loss_zipped = parallel(
                    delayed(compute_for_split)(
                        adj, split_idx_test(split), n, th, pisces_kwargs=pisces_kwargs
                    )
                    for split in range(n_splits)
                )
                modu_split, logllh_split = map(np.array, zip(*loss_zipped))
                modu = sum(modu_split)
                logllh = sum(logllh_split)
        else:
            for split in range(n_splits):
                modu_split, logllh_split = compute_for_split(
                    adj, split_idx_test(split), n, th, pisces_kwargs=pisces_kwargs
                )
                modu = modu + modu_split
                logllh = logllh + logllh_split

        if self.verbose:
            log(
                f"Cross validation, alpha={alpha[0,0]}::: "
                f"modularity:{modu}, loglikelihood:{logllh}"
            )

        return modu, logllh


if __name__ == "__main__":
    # One easy example for PisCES.
    from dypoces.dcbm import MuSDynamicDCBM

    model_dcbm = MuSDynamicDCBM(
        n=200,
        k=5,
        p_in=(0.3, 0.35),
        p_out=0.1,
        time_horizon=5,
        r_time=0.2,
        num_subjects=1,
        r_subject=0.5,
    )
    adj_ms_series, z_ms_series = model_dcbm.simulate_ms_dynamic_dcbm(case=0)
    pisces = PisCES(verbose=True)
    from sklearn.metrics.cluster import adjusted_rand_score

    for i in range(z_ms_series.shape[0]):
        pisces.fit(adj_ms_series[i, :, :], degree_correction=False)
        z_predicted = pisces.predict(alpha=0.1 * np.ones((z_ms_series.shape[1], 2)))
        for j in range(z_ms_series.shape[1]):
            print(adjusted_rand_score(z_predicted[j], z_ms_series[i, j, :]))
