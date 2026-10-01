import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax, xlogy


class SOMOLP:
    def __init__(self, R, gamma=1.0, lam=0.1, max_iters=1000, tol=1e-4, pca_scale=2.0):
        self.R = np.asarray(R, dtype=float)
        self.gamma = gamma
        self.lam = lam
        self.max_iters = max_iters
        self.tol = tol
        self.pca_scale = pca_scale

        self.W = None
        self.P = None
        self.V = None
        self.history = []
        self.n_iter = 0

    def _init_pca(self, X):
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        R_c = self.R - self.R.mean(axis=0, keepdims=True)
        R_c /= np.maximum(np.abs(R_c).max(axis=0, keepdims=True), 1e-12)
        k = min(self.R.shape[1], X.shape[1])
        S = self.pca_scale * (s[:k] / np.sqrt(X.shape[0]))[None, :]
        self.W = mu + (R_c[:, :k] * S) @ Vt[:k]
        self.P = softmax(-cdist(X, self.W, "sqeuclidean") / self.lam, axis=1)

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self._init_pca(X)
        self.history = []
        prev_obj = None

        for _ in range(self.max_iters):
            self.V = self.P @ self.R

            den = self.P.sum(axis=0, keepdims=True).T
            np.divide(self.P.T @ X, den, out=self.W, where=den > 0)

            local_cost = (
                cdist(X, self.W, "sqeuclidean")
                + self.gamma * cdist(self.V, self.R, "sqeuclidean")
            )
            self.P = softmax(-local_cost / self.lam, axis=1)

            obj = np.sum(self.P * local_cost) + self.lam * np.sum(xlogy(self.P, self.P))
            self.history.append(obj)

            if prev_obj is not None and abs(obj - prev_obj) / max(1.0, abs(prev_obj)) <= self.tol:
                break
            prev_obj = obj

        self.n_iter = len(self.history)
        return self
