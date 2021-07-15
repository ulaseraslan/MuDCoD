import sys
import copy
import numpy as np

sys.path.append("../")
from utils.CESmethods import eigen_complete, wLoss
from PisCES.PisCES import pisces


def pisces_cv(
    A,
    alphalist=[0.05, 1],
    degree_correction=True,
    K_max=None,
    n_iter=50,
    K_fold=5,
    verbose=False,
):
    """
    This is a function for cross validation of PisCES method

    Parameters
    ----------
    A
            adjacency matrices with dimention N*N*T, N is the number of nodes and T
            is the number of networks
    degree_correction
            'True' for degree correction
    alpha
            tuning parameter, default as 0.1
    K_max
            maximum number of communities, degault as N/10
    n_iter
            number of iteration of pisces, default is 10
    K_fold
            number of folds in cross validation
    alphalist
            possible alpha to choose from

    Returns
    -------
    modu
            modularity of different alpha
    logllh
            log likelihood of different alpha
    """
    S = A.shape
    T = S[2]
    N = S[1]

    if K_max is None:
        K_max = N // 10

    idxN = np.arange(N)
    idx = np.c_[np.repeat(idxN, idxN.shape), np.tile(idxN, idxN.shape)]
    r = np.random.choice(N ** 2, size=N ** 2)

    cvidx = np.empty((N, N, K_fold))
    A_train = np.zeros((N, N, T, K_fold))
    A_train2 = np.zeros((N, N, T, K_fold))

    for k in range(K_fold):
        ## print(f"Running eigen completion for k={k}.")
        for t in range(T):
            test = r[k * (N ** 2 // K_fold) : (k + 1) * (N ** 2 // K_fold) - 1]

            cvidxtemp = np.zeros((N, N))
            cvidxtemp[idx[test, 0], idx[test, 1]] = 1
            cvidxtemp = np.triu(cvidxtemp) + np.triu(cvidxtemp).T
            cvidx[:, :, k] = cvidxtemp

            At = copy.deepcopy(A[:, :, t])
            At[idx[test, 0], idx[test, 1]] = 0
            At = np.triu(At) + np.triu(At).T
            A_train[:, :, t, k] = At
            A_train2[:, :, t, k] = eigen_complete(At, cvidxtemp, 10, 10)

    la = len(alphalist)
    modu = np.zeros(la)
    logllh = np.zeros(la)

    for a in range(la):
        alpha = np.ones((T, 2)) * alphalist[a]
        Z = np.zeros((N, T, K_fold), dtype=int)

        for k in range(K_fold):
            Z[:, :, k] = pisces(
                A_train2[:, :, :, k], degree_correction, alpha, K_max, n_iter
            )

            if verbose:
                print(f"Cross validation, alpha={alphalist[a]}, fold={k}.")

            for t in range(T):
                modu[a] = modu[a] + wLoss(
                    A[:, :, t], A_train[:, :, t, k], Z[:, t, k], 1, cvidx[:, :, k]
                )
                logllh[a] = logllh[a] + wLoss(
                    A[:, :, t], A_train[:, :, t, k], Z[:, t, k], 2, cvidx[:, :, k]
                )

        if verbose:
            print(f"modularity for alpha={alphalist[a]}: {modu[a]}")
            print(f"loglikelihood for alpha={alphalist[a]}: {logllh[a]}\n")

    return modu, logllh


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

    # Run PisCES_cv
    alphalist = [0.001, 0.01, 0.1]

    modu, logllh = pisces_cv(A, alphalist=alphalist, verbose=True)
