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


def pisces(A, degree_correction=True, alpha=None, K_max=None, n_iter=50, verbose=False):
    """
    Parameters
    ----------
    A
        adjacency matrices with dimention N*N*T,
        N is the number of nodes and T is the number of networks.
    degree_correction
        'True' for degree correction.
    alpha
        tuning parameter, default as 0.1.
    K_max
        maximum number of communities, degault as N/10.
    n_iter
        number of iteration of pisces, default is 50.

    Returns
    -------
    Z: community prediction for each time point.
    """
    S = A.shape
    T = S[2]
    N = S[1]  # or S[0]

    if alpha is None:
        alpha = 0.1 * np.ones((T, 2))
    if K_max is None:
        K_max = N // 10

    if verbose:
        print("PisCES")
        print(f"alpha: {alpha[0,0]}")
        print(f"K_max: {K_max}")
        print(f"n_iter: {n_iter}")
        print("\n")

    Z = np.zeros((T, N))
    k = (np.zeros(T) + K_max).astype(int)
    obj = np.zeros((n_iter))
    ## obj2 = np.zeros((n_iter))
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

    for iter in range(n_iter):
        V_old = copy.deepcopy(V)
        for t in range(T):
            if t == 0:
                At = A[:, :, t]
                X = V_old[:, : k[t + 1], t + 1]
                S = At + alpha[t, 1] * X @ X.T
                k[t] = choose_k(S, D[:, :, t], K_max, At)
                _, V[:, : k[t], t] = eigs(At, k=k[t], which="LM")
                eig_val = eigvals(V[:, : k[t], t].T @ V_old[:, : k[t], t])
                obj[iter] = obj[iter] + (np.sum(np.abs(eig_val), axis=0))

            elif t == T - 1:
                At = A[:, :, t]
                X = V_old[:, : k[t - 1], t - 1]
                S = At + alpha[t, 0] * X @ X.T
                k[t] = choose_k(S, D[:, :, t], K_max, At)
                _, V[:, : k[t], t] = eigs(S, k=k[t], which="LM")
                eig_val = eigvals(V[:, : k[t], t].T @ V_old[:, : k[t], t])
                obj[iter] = obj[iter] + (np.sum(np.abs(eig_val), axis=0))

            else:
                At = A[:, :, t]
                X1 = V_old[:, : k[t - 1], t - 1]
                X2 = V_old[:, : k[t + 1], t + 1]
                S = At + (alpha[t, 0] * X1 @ X1.T) + (alpha[t, 1] * X2 @ X2.T)
                k[t] = choose_k(S, D[:, :, t], K_max, At)
                _, V[:, : k[t], t] = eigs(S, k=k[t], which="LM")
                eig_val = eigvals(V[:, : k[t], t].T @ V_old[:, : k[t], t])
                obj[iter] = obj[iter] + np.sum(np.abs(eig_val), axis=0)

        if verbose:
            print(f"Value of objective funciton: {obj[iter]}, at iteration {iter+1}.")

        if iter > 1 and abs(obj[iter] - obj[iter - 1]) < 0.001:
            break

    if obj[iter] - obj[iter - 1] >= 0.001:
        warnings.warn("PisCES does not converged!", RuntimeWarning)
        print(f"PisCES does not converge for alpha={alpha[1, 1]}.")
        print("Please try a smaller alpha.")

    Z = np.empty((N, T), dtype=int)
    for t in range(T):
        kmeans = KMeans(n_clusters=k[t])
        Z[:, t] = kmeans.fit_predict(V[:, : k[t], t], k[t])

    return Z


if __name__ == "__main__":
    # One easy example for PisCES.
    # Simulate adjacency matrices.
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

    # Run PisCES with alpha = 0.1.
    Z = pisces(
        A, degree_correction=True, alpha=10 ** (-1) * np.ones((T, 2)), verbose=True
    )

    from collections import Counter

    for i in range(Z.shape[1]):
        C_dist = dict(sorted(Counter(Z[:, i]).items()))
        print(C_dist)
