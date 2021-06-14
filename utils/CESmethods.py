import copy
import numpy as np

from numpy.linalg import inv, eigvals, svd
from scipy.linalg import sqrtm
from scipy.special import comb


EPS = 10 ** (-10)


def wLoss(adjTest, adjTrain, zhat, opt, cvidx):
    """
    Calculate the modularity or loglikelihood

    Parameters
    ----------
    Adjtest
            test matrix with dimention N*N; training edges with value 0
    Adjtrain
            training matrix with dimention N*N; test edges with value 0
    zhat
            estimated community assignment with dimension 1*N
    opt
            modulatiry for 1 and loglikelihood for 2
    cvidx
            N*N marix indicates the index of test edges: 1 for test and 0 for
            training

    Returns
    -------
    loss
            modulatiry or loglikelihood on test data
    """
    k = np.max(zhat)
    n = len(zhat)

    loss = 0
    minvalue = 0

    maxval = np.max(adjTrain)
    binA = copy.deepcopy(adjTrain)

    kts = np.sum(binA, axis=0)
    W = np.sum(kts, axis=0)

    if opt == 1:
        row_idx, col_idx = np.nonzero(cvidx == 0)
        hval, _ = np.histogram(col_idx, bins=n)
        ne = np.sum(hval, axis=0)

        for k1 in range(n):
            for k2 in range(n):
                if (cvidx[k1, k2] > 0 and zhat[k1] == zhat[k2]) and k1 != k2:
                    loss = loss + (
                        adjTest[k1, k2]
                        - (kts[k1] / hval[k1] * kts[k2] / hval[k2]) / W * ne
                    )
    else:
        hat0 = np.zeros((k + 1, k + 1), dtype=int)
        theta = np.zeros((n))

        for i in range(k + 1):
            for j in range(k + 1):
                aK = binA[:, zhat == j][zhat == i, :]
                ## aJ = adjTrain[zhat == i, zhat == j]
                ## ajx = cvidx[zhat == i, zhat == j];
                hat0[i, j] = np.sum(aK)

        for k1 in range(n):
            kk = zhat[k1]
            theta[k1] = kts[k1] / np.sum(hat0[kk, :], axis=0)

        for k1 in range(n):
            for k2 in range(n):
                if cvidx[k1, k2] > 0:
                    prob = (5 / 4) * theta[k1] * theta[k2] * hat0[zhat[k1], zhat[k2]]
                    ## prob = theta[k1] * theta[k2] * hat0[zhat[k1], zhat[k2]]

                    if prob == 0 or np.isnan(prob):
                        prob = 10 ** (-5)
                    if prob >= 1:
                        prob = 1 - 10 ** (-5)

                    loss = (
                        loss
                        - np.log(prob) * int((adjTest[k1, k2] > (0.7 ** 6)))
                        - np.log(1 - prob) * int((adjTest[k1, k2] <= (0.7 ** 6)))
                    )
        loss = loss / W

    return loss


def eigen_complete(A, cvidx, epsilon, k):
    M = np.zeros(A.shape)
    while True:
        A2 = (A * (1 - cvidx)) + (M * cvidx)
        u, s, vh = svd(A2)
        s[k:] = 0
        M2 = u @ np.diag(s) @ vh
        M2 = np.where(M2 < 0, 0, M2)
        M2 = np.where(M2 > 1, 1, M2)
        dn = np.sqrt(np.sum((M - M2) ** 2))
        if dn < epsilon:
            break
        else:
            M = M2

    return M


def choose_k(A, D, Kmax, At):
    """
    Method of choosing number of modules K

    Parameters
    ----------
    A
        smoothed adjacency matrix from iteration of PisCES with dimension N*N
    D
        degree matrix with dimention N*N
    Kmax
        then maximum number of modules
    At
        original adjacency matrix with dimension N*N

    Returns
    -------
    K
        number of modules
    """
    n = A.shape[0]

    if np.array_equal(D, np.eye(n)):
        Da = np.diag(np.sum(A, axis=0) + EPS)
        sqinvDa = sqrtm(inv(Da))
        erw = eigvals(D - (sqinvDa @ A @ sqinvDa))
    else:
        erw = eigvals(np.eye(n) - A)

    erw = np.sort(erw)
    gaprw = erw[1 : Kmax + 1] - erw[:Kmax]
    sqD = sqrtm(D)
    pin = np.sum(sqD @ At @ sqD) / comb(n, 2) / 2
    threshold = 3.5 / pin ** (0.58) / n ** (1.15)

    idx = np.nonzero(gaprw > threshold)[0]
    if idx.shape[0] == 0:
        K = 1
    else:
        K = max(idx) + 1

    return int(K)
