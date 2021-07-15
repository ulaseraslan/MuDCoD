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


def muspces(
    A,
    degree_correction=True,
    alpha=None,
    beta=None,
    K_max=None,
    n_iter=50,
    verbose=False,
):
    """
    Parameters
    ----------
    A
        adjacency matrices with dimention Ns,N*N*T,
        Ns is the number of subjects,
        N is the number of nodes and T is the number of networks.
    degree_correction
        'True' for degree correction.
    alpha
        tuning parameter along time axis, default as 0.1.
    beta
        tuning parameter along subject axis, default as 0.1.
    K_max
        maximum number of communities, degault as N/10.
    n_iter
        number of iteration of muspces, default is 50.

    Returns
    -------
    Z: community prediction for each time point.
    """
    S = A.shape
    T = S[3]
    N = S[1]  # or S[2]
    Ns = S[0]

    if alpha is None:
        alpha = 0.1 * np.ones((T, 2))
    if beta is None:
        beta = 0.1 * np.ones(T)
    if K_max is None:
        K_max = N // 10

    if verbose:
        print("MuSPCES")
        print(f"alpha: {alpha[0,0]}")
        print(f"beta: {beta[0]}")
        print(f"K_max: {K_max}")
        print(f"n_iter: {n_iter}")
        print("\n")

    Z = np.zeros((Ns, T, N))
    k = (np.zeros((Ns, T)) + K_max).astype(int)
    obj = np.zeros((n_iter))
    ## obj2 = np.zeros((n_iter))
    V = np.zeros((Ns, N, K_max, T))
    D = copy.deepcopy(A)

    if degree_correction:
        for t in range(T):
            for sbj in range(Ns):
                At = A[sbj, :, :, t]
                Dg = np.diag(np.sum(np.abs(At), axis=0) + EPS)
                sqinvDg = sqrtm(inv(Dg))
                A[sbj, :, :, t] = sqinvDg @ At @ sqinvDg
                D[sbj, :, :, t] = Dg
    else:
        for t in range(T):
            for sbj in range(Ns):
                D[sbj, :, :, t] = np.eye(N)

    # Initialization of k, V.
    for t in range(T):
        for sbj in range(Ns):
            At = A[sbj, :, :, t]
            k[sbj, t] = choose_k(At, D[sbj, :, :, t], K_max, At)
            _, V[sbj, :, : k[sbj, t], t] = eigs(At, k=k[sbj, t], which="LM")

    for iter in range(n_iter):
        V_old = copy.deepcopy(V)
        for t in range(T):

            X_t = V_old[:, :, :, t]
            XT_t = np.swapaxes(X_t, 1, 2)
            muU_t = np.mean(X_t @ XT_t, axis=0)

            for sbj in range(Ns):
                if t == 0:
                    At = A[sbj, :, :, t]
                    X_tn = V_old[sbj, :, : k[sbj, t + 1], t + 1]
                    S = At + alpha[t, 1] * X_tn @ X_tn.T + beta[t] * muU_t
                    k[sbj, t] = choose_k(S, D[sbj, :, :, t], K_max, At)
                    _, V[sbj, :, : k[sbj, t], t] = eigs(At, k=k[sbj, t], which="LM")
                    eig_val = eigvals(
                        V[sbj, :, : k[sbj, t], t].T @ V_old[sbj, :, : k[sbj, t], t]
                    )
                    obj[iter] = obj[iter] + (np.sum(np.abs(eig_val), axis=0))

                elif t == T - 1:
                    At = A[sbj, :, :, t]
                    X_tp = V_old[sbj, :, : k[sbj, t - 1], t - 1]
                    S = At + alpha[t, 0] * X_tp @ X_tp.T + beta[t] * muU_t
                    k[sbj, t] = choose_k(S, D[sbj, :, :, t], K_max, At)
                    _, V[sbj, :, : k[sbj, t], t] = eigs(S, k=k[sbj, t], which="LM")
                    eig_val = eigvals(
                        V[sbj, :, : k[sbj, t], t].T @ V_old[sbj, :, : k[sbj, t], t]
                    )
                    obj[iter] = obj[iter] + (np.sum(np.abs(eig_val), axis=0))

                else:
                    At = A[sbj, :, :, t]
                    X_tp = V_old[sbj, :, : k[sbj, t - 1], t - 1]
                    X_tn = V_old[sbj, :, : k[sbj, t + 1], t + 1]
                    S = (
                        At
                        + (alpha[t, 0] * X_tp @ X_tp.T)
                        + (alpha[t, 1] * X_tn @ X_tn.T)
                        + beta[t] * muU_t
                    )
                    k[sbj, t] = choose_k(S, D[sbj, :, :, t], K_max, At)
                    _, V[sbj, :, : k[sbj, t], t] = eigs(S, k=k[sbj, t], which="LM")
                    eig_val = eigvals(
                        V[sbj, :, : k[sbj, t], t].T @ V_old[sbj, :, : k[sbj, t], t]
                    )
                    obj[iter] = obj[iter] + np.sum(np.abs(eig_val), axis=0)

        if verbose:
            print(f"Value of objective funciton: {obj[iter]}, at iteration {iter+1}.")

        if iter > 1 and abs(obj[iter] - obj[iter - 1]) < 0.001:
            break

    if obj[iter] - obj[iter - 1] >= 0.001:
        warnings.warn("MuSPCES does not converged!", RuntimeWarning)
        print(f"MuSPCES does not converge for alpha={alpha[1, 1]}, beta={beta[0]}.")
        print("Please try a smaller alpha/beta.")

    Z = np.empty((Ns, N, T), dtype=int)
    for t in range(T):
        for sbj in range(Ns):
            kmeans = KMeans(n_clusters=k[sbj, t])
            Z[sbj, :, t] = kmeans.fit_predict(V[sbj, :, : k[sbj, t], t], k[sbj, t])
    return Z


if __name__ == "__main__":
    # One easy example for FooCES.
    # Simulate adjacency matrices.
    T = 6
    Ns = 4
    K = 4
    N = 200
    N = (N // K) * K

    P = np.eye(K) * 0.3 + np.ones(K) * 0.1
    A = np.zeros((Ns, N, N, T))
    pz = np.tile(P, (N // K, N // K))

    for t in range(T):
        for sbj in range(Ns):
            r = np.random.default_rng().uniform(0, 1, (N, N))
            At = np.greater_equal(pz, r).astype(float)
            A[sbj, :, :, t] = np.triu(At, 1) + np.triu(At, 1).T

    # Run FooCES with alpha = 0.1, beta=0.1.
    alpha = 10 ** (-1) * np.ones((T, 2))
    beta = 1 * 10 ** (-1) * np.ones(T)

    Z = muspces(A, degree_correction=True, alpha=alpha, beta=beta, verbose=True)

    from collections import Counter

    for Z_sbj in Z:
        for i in range(Z_sbj.shape[1]):
            C_dist = dict(sorted(Counter(Z_sbj[:, i]).items()))
            print(C_dist)
