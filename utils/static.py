import sys
import copy
import warnings
import numpy as np

from numpy.linalg import inv, eigvals
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans

sys.path.append("../")
from utils.CESmethods import choose_k


EPS = 10 ** (-10)


def static_clustering(A, degree_correction=True, K_max=None, verbose=False):
    """
    Parameters
    ----------
    A
        adjacency matrices with dimention N*N*T,
        N is the number of nodes and T is the number of networks.
    degree_correction
        'True' for degree correction.
    K_max
        maximum number of communities, degault as N/10.

    Returns
    -------
    Z: community prediction for each time point.
    """
    S = A.shape
    T = S[2]
    N = S[1]  # or S[0]

    if K_max is None:
        K_max = N // 10

    if verbose:
        print("Static.")
        print(f"K_max: {K_max}")
        print(f"\n")

    Z = np.zeros((T, N))
    k = (np.zeros(T) + K_max).astype(int)
    V = np.zeros((N, K_max, T))
    D = copy.deepcopy(A)

    if degree_correction:
        for t in range(T):
            At = A[:, :, t]
            Dg = np.diag(np.sum(np.abs(At), axis=0) + EPS)
            sqinvDg = sqrtm(inv(Dg))
            A[:, :, t] = sqinvDg @ At @ sqinvDg
            D[:, :, t] = Dg
    else:
        for t in range(T):
            D[:, :, t] = np.eye(N)

    # Initialization of k, V.
    for t in range(T):
        At = A[:, :, t]
        k[t] = choose_k(At, D[:, :, t], K_max, At)
        _, V[:, : k[t], t] = eigs(At, k=k[t], which="LM")

    Z = np.empty((N, T), dtype=int)
    for t in range(T):
        kmeans = KMeans(n_clusters=k[t])
        Z[:, t] = kmeans.fit_predict(V[:, : k[t], t], k[t])

    return Z


if __name__ == "__main__":
    T = 6
    K = 4
    N = 200
    N = (N // K) * K

    P = np.eye(K) * 0.3 + np.ones(K) * 0.1
    A = np.zeros((N, N, T))
    pz = np.tile(P, (N // K, N // K))

    for t in range(T):
        r = np.random.default_rng().uniform(0, 1, (N, N))
        At = np.greater_equal(pz, r).astype(float)
        A[:, :, t] = np.triu(At, 1) + np.triu(At, 1).T

    Z = static(A, degree_correction=True, verbose=True)

    from collections import Counter

    for i in range(Z.shape[1]):
        C_dist = dict(sorted(Counter(Z[:, i]).items()))
        print(C_dist)
