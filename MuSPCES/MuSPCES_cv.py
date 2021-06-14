import sys
import copy
import numpy as np

sys.path.append("../")
from utils.CESmethods import choose_k, eigen_complete, wLoss
from MuSPCES.MuSPCES import muspces


def muspces_cv(
    A,
    alphalist=[0.05, 1],
    betalist=[0.05, 1],
    degree_correction=True,
    K_max=None,
    T_n=50,
    K_fold=5,
    verbose=False,
):
    """
    This is a function for cross validation of PisCES method

    Parameters
    ----------
    A
            adjacency matrices with dimention Ns,N*N*T,
            Ns is the number of subjects,
            N is the number of nodes and T is the number of networks
    degree_correction
            'True' for degree correction
    K_max
            maximum number of communities, degault as N/10
    T_n
            number of iteration of pisces, default is 10
    K_fold
            number of folds in cross validation
    alphalist
            possible alpha to choose from

    betalist
            possible beta to choose from

    Returns
    -------
    modu
            modularity of different alpha
    logllh
            log likelihood of different alpha
    """
    S = A.shape
    T = S[3]
    N = S[1]  # or S[2]
    Ns = S[0]

    if K_max == None:
        K_max = N // 10

    idxN = np.arange(N)
    idx = np.c_[np.repeat(idxN, idxN.shape), np.tile(idxN, idxN.shape)]
    r = np.random.choice(N ** 2, size=N ** 2)

    cvidx = np.empty((N, N, K_fold))
    A_train = np.zeros((Ns, N, N, T, K_fold))
    A_train2 = np.zeros((Ns, N, N, T, K_fold))

    for k in range(K_fold):
        ## print(f"Running eigen completion for k={k}.")
        for t in range(T):
            for sbj in range(Ns):
                test = r[k * (N ** 2 // K_fold) : (k + 1) * (N ** 2 // K_fold) - 1]

                cvidxtemp = np.zeros((N, N))
                cvidxtemp[idx[test, 0], idx[test, 1]] = 1
                cvidxtemp = np.triu(cvidxtemp) + np.triu(cvidxtemp).T
                cvidx[:, :, k] = cvidxtemp

                At = copy.deepcopy(A[sbj, :, :, t])
                At[idx[test, 0], idx[test, 1]] = 0
                At = np.triu(At) + np.triu(At).T
                A_train[sbj, :, :, t, k] = At
                A_train2[sbj, :, :, t, k] = eigen_complete(At, cvidxtemp, 10, 10)

    la = len(alphalist)
    lb = len(betalist)
    modu = np.zeros((la, lb))
    logllh = np.zeros((la, lb))

    for a in range(la):
        for b in range(lb):
            alpha = np.ones((T, 2)) * alphalist[a]
            beta = np.ones(T) * betalist[b]
            Z = np.zeros((Ns, N, T, K_fold), dtype=int)

            for k in range(K_fold):
                Z[:, :, :, k] = muspces(
                    A_train2[:, :, :, :, k], degree_correction, alpha, beta, K_max, T_n
                )

                if verbose:
                    print(
                        f"Cross validation, alpha={alphalist[a]}, beta={betalist[b]} fold={k}."
                    )

                for t in range(T):
                    for sbj in range(Ns):
                        modu[a, b] = modu[a, b] + wLoss(
                            A[sbj, :, :, t],
                            A_train[sbj, :, :, t, k],
                            Z[sbj, :, t, k],
                            1,
                            cvidx[:, :, k],
                        )
                        logllh[a, b] = logllh[a, b] + wLoss(
                            A[sbj, :, :, t],
                            A_train[sbj, :, :, t, k],
                            Z[sbj, :, t, k],
                            2,
                            cvidx[:, :, k],
                        )

            if verbose:
                print(
                    f"Total modularity for alpha={alphalist[a]}, beta={betalist[b]}: {modu[a,b]}"
                )
                print(
                    f"Total loglikelihood for alpha={alphalist[a]}, beta={betalist[b]}: {logllh[a,b]}\n"
                )

    return modu, logllh


if __name__ == "__main__":
    # One easy example for PisCES.
    # Simulate adjacency matrices.
    T = 6
    Ns = 2
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

    # Run PisCES_cv
    alphalist = [0.001, 0.01, 0.1]
    betalist = [0.00001, 0.001, 0.005]

    modu, logllh = muspces_cv(A, alphalist=alphalist, betalist=betalist, verbose=True)
