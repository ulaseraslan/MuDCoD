import numpy as np

from copy import deepcopy
from numpy.random import default_rng


class DCBM:
    def __init__(self, n=None, k=None, p_in=None, p_out=None):
        self.n = n
        self.k = k
        self.p_in = p_in
        self.p_out = p_out
        if None in [n, k, p_in, p_out]:
            raise ValueError

    @staticmethod
    def loss(adj_test, adj_train, z_hat, cvidx, opt="logllh"):
        """
        Calculate the modularity or loglikelihood

        Parameters
        ----------
        adj_test
                test matrix with dimention (n,n); training edges with value 0
        adj_train
                training matrix with dimention (n,n); test edges with value 0
        z_hat
                estimated community assignment with dimension (1,n)
        opt
                'modu' for modulatiry and 'logllh' loglikelihood
        cvidx
                (n,n) marix indicates the index of test edges: 1 for test and 0 for
                training

        Returns
        -------
        loss
                modulatiry or loglikelihood on test data
        """
        k = np.max(z_hat)
        n = len(z_hat)

        loss = 0
        adj_train_dc = deepcopy(adj_train)

        kts = np.sum(adj_train_dc, axis=0)
        w = np.sum(kts, axis=0)

        if opt == "modu":
            row_idx, col_idx = np.nonzero(cvidx == 0)
            hval, _ = np.histogram(col_idx, bins=n)
            ne = np.sum(hval, axis=0)

            for k1 in range(n):
                for k2 in range(n):
                    if (cvidx[k1, k2] > 0 and z_hat[k1] == z_hat[k2]) and k1 != k2:
                        loss = loss + (
                            adj_test[k1, k2]
                            - (kts[k1] / hval[k1] * kts[k2] / hval[k2]) / w * ne
                        )

        elif opt == "logllh":
            hat0 = np.zeros((k + 1, k + 1), dtype=int)
            theta = np.zeros((n))

            for i in range(k + 1):
                for j in range(k + 1):
                    a_k = adj_train_dc[:, z_hat == j][z_hat == i, :]
                    ## a_j = adj_train[z_hat == i, z_hat == j]
                    ## ajx = cvidx[z_hat == i, z_hat == j];
                    hat0[i, j] = np.sum(a_k)

            for k1 in range(n):
                kk = z_hat[k1]
                theta[k1] = kts[k1] / np.sum(hat0[kk, :], axis=0)

            for k1 in range(n):
                for k2 in range(n):
                    if cvidx[k1, k2] > 0:
                        prob = (
                            (5 / 4) * theta[k1] * theta[k2] * hat0[z_hat[k1], z_hat[k2]]
                        )
                        ## prob = theta[k1] * theta[k2] * hat0[z_hat[k1], z_hat[k2]]

                        if prob == 0 or np.isnan(prob):
                            prob = 10 ** (-5)
                        if prob >= 1:
                            prob = 1 - 10 ** (-5)

                        loss = (
                            loss
                            - np.log(prob) * int((adj_test[k1, k2] > (0.7 ** 6)))
                            - np.log(1 - prob) * int((adj_test[k1, k2] <= (0.7 ** 6)))
                        )
            loss = loss / w

        else:
            raise ValueError(
                f"Given option {opt} for loss value computation is unkown."
            )
        return loss

    def _get_random_z(self):
        rng = default_rng()
        k = self.k
        n = self.n
        z_init = np.nonzero(rng.multinomial(1, [1 / k] * k, size=n))[1]
        return z_init

    def _z_evolve(self, z_old, r):
        rng = default_rng()
        assert z_old.shape[0] == self.n
        n = self.n
        k = self.k
        e_r = rng.binomial(n, r)
        z_tn = np.nonzero(rng.multinomial(1, [1 / k] * k, size=n))[1]
        z_new = np.where(e_r == 1, z_tn, z_old)
        return z_new

    def get_adjacency(self, z, B, theta):
        rng = default_rng()
        n = self.n
        assert z.shape[0] == self.n

        adj = np.zeros((n, n), dtype=int)
        p = np.empty((n, n))

        bz = B[np.ix_(z[:], z[:])]
        p[:, :] = theta[np.newaxis, :] * theta[:, np.newaxis] * bz
        adj_triu = rng.binomial(1, p)
        adj[:, :] = np.triu(adj_triu, 1) + np.triu(adj_triu, 1).T

        return adj

    def dcbm(self, z):
        rng = default_rng()
        assert z.shape[0] == self.n
        n = self.n
        k = self.k
        p_in, p_out = self.p_in, self.p_out
        if p_in[1] < p_in[0]:
            raise ValueError("p_in[1] should be greater than or equal to p_in[0].")
        if p_in[1] < p_out:
            raise ValueError("p_in[1] should be greater than p_out.")

        B = np.full((k, k), p_out)
        B[np.diag_indices(k)] = rng.uniform(p_in[0], p_in[1], k)

        if p_in[1] < (1 / (1.5 ** 2)):
            gamma1 = 1
            gamma0 = 0.5
        elif p_in[1] < (1 / (1.25 ** 2)):
            gamma1 = 0.75
            gamma0 = 0.5
        else:
            gamma1 = 0.9
            gamma0 = 0.1 - np.finfo(np.float32).eps

        theta = gamma1 * (np.random.permutation(n) / n) + gamma0

        adj = self.get_adjacency(z, B, theta)

        return adj, z


class DynamicDCBM(DCBM):
    def __init__(
        self, n=None, k=None, p_in=None, p_out=None, time_horizon=None, r_time=None
    ):
        self.time_horizon = time_horizon
        self.r_time = r_time
        if None in [n, k, p_in, p_out, time_horizon, r_time]:
            raise ValueError
        super().__init__(n, k, p_in, p_out)

    def dynamic_dcbm(self, z):
        z_new = self._z_evolve(z, self.r_time)
        adj, _ = self.dcbm(z_new)
        return adj, z_new

    def simulate_dynamic_dcbm(self):
        n = self.n
        th = self.time_horizon

        adj_series = np.empty((th, n, n), dtype=int)
        z_series = np.empty((th, n), dtype=int)

        z_init = self._get_random_z()
        adj_series[0, :, :], z_series[0, :] = self.dcbm(z_init)

        for t in range(1, th):
            adj_series[t, :, :], z_series[t, :] = self.dynamic_dcbm(z_series[t - 1, :])

        return adj_series, z_series


class MuSDynamicDCBM(DynamicDCBM):
    def __init__(
        self,
        n=None,
        k=None,
        p_in=None,
        p_out=None,
        time_horizon=None,
        r_time=None,
        num_subjects=None,
        r_subject=None,
    ):
        self.num_subjects = num_subjects
        self.r_subject = r_subject
        if None in [n, k, p_in, p_out, time_horizon, r_time, num_subjects, r_subject]:
            raise ValueError
        super().__init__(n, k, p_in, p_out, time_horizon, r_time)

    def mus_dynamic_dcbm(self, z):
        z_new = self._z_evolve(z, self.r_subject)
        adj, z_new = self.dcbm(z_new)
        return adj, z_new

    def simulate_ms_dynamic_dcbm(self, case=3):
        n = self.n
        th = self.time_horizon
        num_sbj = self.num_subjects
        adj_ms_series = np.empty((num_sbj, th, n, n), dtype=int)
        z_ms_series = np.empty((num_sbj, th, n), dtype=int)

        # Totally independent subjects, evolve independently.
        if case == 0:
            for sbj in range(num_sbj):
                (
                    adj_ms_series[sbj, :, :, :],
                    z_ms_series[sbj, :, :],
                ) = self.simulate_dynamic_dcbm()
        # Subjects are siblings at time 0, then they evolve independently.
        elif case == 1:
            z_init = self._get_random_z()
            adj_ms_series[0, 0, :, :], z_ms_series[0, 0, :] = self.dcbm(z_init)
            for sbj in range(1, num_sbj):
                (
                    adj_ms_series[sbj, 0, :, :],
                    z_ms_series[sbj, 0, :],
                ) = self.mus_dynamic_dcbm(z_init)
            for sbj in range(0, num_sbj):
                for t in range(1, th):
                    (
                        adj_ms_series[sbj, t, :, :],
                        z_ms_series[sbj, t, :],
                    ) = self.dynamic_dcbm(z_ms_series[sbj, t - 1, :])
        elif case == 2:
            z_init = self._get_random_z()
            adj_ms_series[0, 0, :, :], z_ms_series[0, 0, :] = self.dcbm(z_init)
            for sbj in range(1, num_sbj):
                (
                    adj_ms_series[sbj, 0, :, :],
                    z_ms_series[sbj, 0, :],
                ) = self.mus_dynamic_dcbm(z_ms_series[sbj - 1, 0, :])
            for sbj in range(0, num_sbj):
                for t in range(1, th):
                    (
                        adj_ms_series[sbj, t, :, :],
                        z_ms_series[sbj, t, :],
                    ) = self.dynamic_dcbm(z_ms_series[sbj, t - 1, :])
        elif case == 3:
            (
                adj_ms_series[0, :, :, :],
                z_ms_series[0, :, :],
            ) = self.simulate_dynamic_dcbm()
            for sbj in range(1, num_sbj):
                for t in range(0, th):
                    (
                        adj_ms_series[sbj, t, :, :],
                        z_ms_series[sbj, t, :],
                    ) = self.mus_dynamic_dcbm(z_ms_series[0, t, :])
        elif case == 4:
            (
                adj_ms_series[0, :, :, :],
                z_ms_series[0, :, :],
            ) = self.simulate_dynamic_dcbm()
            for sbj in range(1, num_sbj):
                for t in range(0, th):
                    (
                        adj_ms_series[sbj, t, :, :],
                        z_ms_series[sbj, t, :],
                    ) = self.mus_dynamic_dcbm(z_ms_series[sbj - 1, t, :])
        else:
            raise ValueError(f"Given case number {case} is not defined.")

        return adj_ms_series, z_ms_series


if __name__ == "__main__":
    T = 3
    k = 4
    n = 100
    r_time = 0.2
    r_subject = 0.2
    p_in = (0.3, 0.4)
    p_out = 0.1
    Ns = 4
    MSDDCBM = MuSDynamicDCBM(n, k, p_in, p_out, T, r_time, Ns, r_subject)
    for i in [0, 1, 2, 3, 4]:
        adj_ms_series, z_ms_series = MSDDCBM.simulate_ms_dynamic_dcbm(case=i)
        ## print(adj_ms_series, z_ms_series)
