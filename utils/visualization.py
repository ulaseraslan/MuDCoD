import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class nxDraw:
    @staticmethod
    def draw_spectral(A, path, **kwargs):
        G = nx.from_numpy_matrix(A)
        nx.draw_spectral(G, **kwargs)
        plt.savefig(path)

    @staticmethod
    def draw_kamada_kawai(A, path, **kwargs):
        G = nx.from_numpy_matrix(A)
        nx.draw_kamada_kawai(G, **kwargs)
        plt.savefig(path)

if __name__ == "__main__":
    n = 500
    p_in = (0.8, 0.9)
    p_out = 0.5
    K = 4

    from dcbm import dcbm

    rng = np.random.default_rng()
    z = np.nonzero(rng.multinomial(1, [1 / K] * K, size=n))[1][np.newaxis, :]
    A = dcbm(z, K, p_in, p_out, batch=1)
    ## draw_spectral(A[0, :, :])
    nxDraw.draw_kamada_kawai(A[0, :, :], "../results/graph.pdf", cmap="tab20", node_color=z[0,:])
