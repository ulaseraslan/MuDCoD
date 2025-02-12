import numpy as np

from numpy.linalg import inv, eigvals, svd
from scipy.linalg import sqrtm
from scipy.special import comb


_eps = 10 ** (-10)


class SpectralClustering:
    def __init__(self, method, verbose=False):
        self.verbose = verbose
        self.method = method.lower()
        assert type(method) == str and method.lower() in ["pisces", "muspces", "static"]
        self._embedding = None
        self._model_order_k = None

    @property
    def embedding(self):
        if self._embedding is None:
            raise ValueError("Embeddings are not computed yet, run 'fit' first.")
        return self._embedding

    @property
    def model_order_k(self):
        if self._model_order_k is None:
            raise ValueError("Model order K is not computed yet, run 'fit' first.")
        return self._model_order_k

    @embedding.setter
    def embedding(self, value):
        if not isinstance(value, np.ndarray):
            raise ValueError("Embeddings must be instance 'np.ndarray'.")
        else:
            self._embedding = value

    @model_order_k.setter
    def model_order_k(self, value):
        if self.method in ["pisces", "muspces"] and not isinstance(value, np.ndarray):
            raise ValueError("Model order K must be instance of 'np.ndarray'.")
        elif self.method in ["static"] and not isinstance(value, int):
            raise ValueError("Model order K must be instance of 'int'.")
        else:
            self._model_order_k = value

    @staticmethod
    def eigen_complete(adj, cvidx, epsilon, k):
        m = np.zeros(adj.shape)
        while True:
            adj_cv = (adj * (1 - cvidx)) + (m * cvidx)
            u, s, vh = svd(adj_cv)
            s[k:] = 0
            m2 = u @ np.diag(s) @ vh
            m2 = np.where(m2 < 0, 0, m2)
            m2 = np.where(m2 > 1, 1, m2)
            dn = np.sqrt(np.sum((m - m2) ** 2))
            if dn < epsilon:
                break
            else:
                m = m2
        return m

    @staticmethod
    def choose_k(adj_smoothed, adj, degree, k_max, opt="empirical"):
        """
        Method of choosing number of modules k

        Parameters
        ----------
        adj_smoothed
            smoothed adjacency matrix from iteration of PisCES with dimension (n,n)
        degree
            degree matrix with dimention (n,n)
        k_max
            then maximum number of modules
        adj
            original adjacency matrix with dimension (n,n)
        opt
            if 'null', then simulate Laplacianized Erdos–Renyi adjacency matrices
            if emprical, use second largest eigengap as the threshold


        Returns
        -------
        k
            number of modules
        """
        n = adj_smoothed.shape[0]
        opt_list = ["null", "empirical"]

        if np.array_equal(degree, np.eye(n)):
            d_a = np.diag(np.sum(adj_smoothed, axis=0) + _eps)
            sqinv_d_a = sqrtm(inv(d_a))
            erw = eigvals(degree - (sqinv_d_a @ adj_smoothed @ sqinv_d_a))
        else:
            erw = eigvals(np.eye(n) - adj_smoothed)

        erw = np.sort(erw)
        gaprw = erw[1 : k_max + 1] - erw[:k_max]
        sq_d = sqrtm(degree)
        pin = np.sum(sq_d @ adj @ sq_d) / comb(n, 2) / 2

        if opt == opt_list[0]:
            threshold = 3.5 / pin ** (0.58) / n ** (1.15)
            idx = np.nonzero(gaprw > threshold)[0]
            if idx.shape[0] == 0:
                k = 1
            else:
                k = np.max(idx) + 1
        elif opt == opt_list[1]:
            k = np.argmax(gaprw[1:]) + 2
        else:
            raise ValueError(
                f"Unkown option {opt} is given for choosing K. Use one of {opt_list}."
            )

        return int(k)
