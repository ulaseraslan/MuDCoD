import numpy as np

from numpy.random import default_rng


_eps = 10 ** (-5)


class DCBM:
    def __init__(self, n=None, k=None, p_in=None, p_out=None):
        self.n = n
        self.k = k
        self.p_in = p_in
        self.p_out = p_out
        if None in [n, k, p_in, p_out]:
            raise ValueError

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
        e_r = rng.binomial(1, r, size=n)
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

        ###
        # This part is not general for DCBMs
        # It only cover a specifiy type of ktsree distribution
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
        ###

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

    def simulate_ms_dynamic_dcbm(self, scenario=3):
        n = self.n
        th = self.time_horizon
        num_sbj = self.num_subjects
        adj_ms_series = np.empty((num_sbj, th, n, n), dtype=int)
        z_ms_series = np.empty((num_sbj, th, n), dtype=int)

        # Totally independent subjects, evolve independently.
        if scenario == 0:
            for sbj in range(num_sbj):
                (
                    adj_ms_series[sbj, :, :, :],
                    z_ms_series[sbj, :, :],
                ) = self.simulate_dynamic_dcbm()
        # Subjects are siblings at time 0, then they evolve independently.
        elif scenario == 1:
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
        elif scenario == 2:
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
        elif scenario == 3:
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
        elif scenario == 4:
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
            raise ValueError(f"Given scenario number {scenario} is not defined.")

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
        adj_ms_series, z_ms_series = MSDDCBM.simulate_ms_dynamic_dcbm(scenario=i)
        ## print(adj_ms_series, z_ms_series)
