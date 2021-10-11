import warnings
import numpy as np

from copy import deepcopy
from numpy.linalg import inv
from scipy.sparse.linalg import eigs
from scipy.linalg import sqrtm
from sklearn.cluster import KMeans

from mudcod.spectral import SpectralClustering
from mudcod.utils.sutils import log


warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

_eps = 10 ** (-10)


class Static(SpectralClustering):
    def __init__(self, verbose=False):
        super().__init__("static", verbose=verbose)

    def fit(self, adj, k_max=None, degree_correction=True):
        """
        Parameters
        ----------
        adj
            adjacency matrices with dimention (n,n),
            n is the number of nodes.
        k_max
            maximum number of communities, default is n/10.
        degree_correction
            degree normalization, default is 'True'.

        Returns
        -------
        embeddings: computed spectral embeddings, with shape (n, k)
        """
        assert type(adj) in [np.ndarray, np.memmap] and adj.ndim == 2

        self.adj = adj.astype(float)
        self.adj_shape = self.adj.shape
        self.n = self.adj_shape[0]  # or adj_shape[1]
        self.degree = deepcopy(adj)
        self.degree_correction = degree_correction

        if k_max is None:
            k_max = self.n // 10
            if self.verbose:
                log(f"k_max is not provided, default value is floor({self.n}/10).")
        if self.verbose:
            log(
                f"Static-fit ~ "
                f"#nodes:{self.n}, "
                f"k-max:{k_max}, "
                f"degree-correction:{degree_correction}"
            )

        if self.degree_correction:
            dg = np.diag(np.sum(np.abs(adj), axis=0) + _eps)
            sqinv_degree = sqrtm(inv(dg))
            self.adj = sqinv_degree @ adj @ sqinv_degree
            self.degree = dg
        else:
            self.degree = np.eye(self.n)

        # initialization of k, v_col.
        v_col = np.zeros((self.n, k_max))
        k = self.choose_k(self.adj, self.adj, self.degree, k_max)
        _, v_col[:, :k] = eigs(self.adj, k=k, which="LM")

        self.embeddings = v_col
        self.model_order_k = k

        return self.embeddings

    def predict(self):
        """
        Parameters
        ----------

        Returns
        -------
        z_series: community prediction for each time point, with shape (n).
        """
        if self.verbose:
            log("Static-predict ~ ")

        kmeans = KMeans(n_clusters=self.model_order_k)
        z = kmeans.fit_predict(self.embeddings[:, : self.model_order_k])

        return z

    def fit_predict(self, adj, k_max=None, degree_correction=True):
        self.fit(adj, k_max=k_max, degree_correction=degree_correction)
        return self.predict()
