import math
import numpy as np

from numpy.random import default_rng


def get_adjacency(z, B, theta, batch=1):
    n = theta.shape[0]

    Adjs = np.zeros((batch, n, n), dtype=int)
    p = np.empty((n, n))

    for i in range(batch):
        rng = default_rng()
        bz = B[np.ix_(z[i, :], z[i, :])]
        p[:, :] = theta[np.newaxis, :] * theta[:, np.newaxis] * bz

        adj = rng.binomial(1, p)
        Adjs[i, :, :] = np.triu(adj, 1) + np.triu(adj, 1).T

    return Adjs


def dcbm(z, K, p_in, p_out, batch=1):
    rng = default_rng()
    n = z.shape[1]

    if p_in[1] < p_in[0]:
        raise ValueError("p_in[1] should be greater than or equal to p_in[0].")
    if p_in[1] < p_out:
        raise ValueError("p_in[1] should be greater than p_out.")

    B = np.full((K, K), p_out)
    B[np.diag_indices(K)] = rng.uniform(p_in[0], p_in[1], K)

    if p_in[1] < (1 / (1.5 ** 2)):
        gamma1 = 1
        gamma0 = 0.5
    elif p_in[1] < (1 / (1.25 ** 2)):
        gamma1 = 1
        gamma0 = 0.25
    else:
        gamma1 = 1
        gamma0 = 0

    theta = gamma1 * (np.random.permutation(n) / n) + gamma0

    Adjs = get_adjacency(z, B, theta, batch=batch)

    return Adjs


def dynamic_dcbm(z, K, r, p_in, p_out, batch=1):
    rng = default_rng()
    n = z.shape[1]

    e_r = rng.binomial(batch, r, n)

    z_tn = np.empty((batch, n), dtype=int)
    for i in range(batch):
        z_tn[i, :] = np.nonzero(np.random.multinomial(1, [1 / K] * K, size=n))[1]

    z_new = np.empty((batch, n), dtype=int)
    for i in range(batch):
        z_new[i, :] = np.where(e_r == 1, z_tn[i, :], z[i, :])

    Adjs = dcbm(z_new, K, p_in, p_out, batch=batch)

    return Adjs, z_new


def simulate_dynamic_dcbm(T, K, n, r, p_in, p_out, batch):
    rng = default_rng()
    Adj_series = np.empty((batch, n, n, T), dtype=int)
    z_series = np.empty((batch, n, T), dtype=int)

    z_init = np.nonzero(np.random.multinomial(1, [1 / K] * K, size=n))[1]
    for i in range(batch):
        z_series[i, :, 0] = z_init

    Adj_series[:, :, :, 0] = dcbm(z_series[:, :, 0], K, p_in, p_out, batch=batch)

    for t in range(1, T):
        Adj_series[:, :, :, t], z_series[:, :, t] = dynamic_dcbm(
            z_series[:, :, t - 1], K, r, p_in, p_out, batch=batch
        )

    return Adj_series, z_series


if __name__ == "__main__":
    T = 3
    K = 4
    n = 100
    r = 0.9
    p_in = (0.3, 0.45)
    p_out = 0.1
    batch = 1

    Adj_series, z_series = simulate_dynamic_dcbm(T, K, n, r, p_in, p_out, batch)

    if __debug__:
        print(Adj_series.shape, z_series.shape)
        for t in range(0, T):
            ## print(Adj_series[:, :, :, t])
            print(z_series[0, :, t])
