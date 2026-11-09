"""Small statistical detectors from the atlas that have no standard sklearn class.

All follow one rule: `score(X)` is higher for more abnormal rows.
"""
import numpy as np


class ZScore:
    def fit(self, X):
        self.m, self.s = X.mean(0), X.std(0) + 1e-12
        return self

    def score(self, X):
        return np.abs((X - self.m) / self.s).max(1)


class IQR:
    def fit(self, X):
        self.q1, self.q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        self.iqr = self.q3 - self.q1 + 1e-12
        return self

    def score(self, X):
        return np.maximum((self.q1 - X) / self.iqr, (X - self.q3) / self.iqr).max(1)


class Mahalanobis:
    def fit(self, X):
        self.m = X.mean(0)
        self.vi = np.linalg.pinv(np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1]))
        return self

    def score(self, X):
        d = X - self.m
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.vi, d), 0))


class KNNDistance:
    def __init__(self, n_neighbors=5):
        self.k = n_neighbors

    def fit(self, X):
        from sklearn.neighbors import NearestNeighbors

        self.nn = NearestNeighbors(n_neighbors=min(self.k, len(X))).fit(X)
        return self

    def score(self, X):
        return self.nn.kneighbors(X)[0][:, -1]


class PCAError:
    def __init__(self, n_components=2):
        self.n = n_components

    def fit(self, X):
        from sklearn.decomposition import PCA

        self.pca = PCA(n_components=min(self.n, X.shape[0], X.shape[1])).fit(X)
        return self

    def score(self, X):
        return ((X - self.pca.inverse_transform(self.pca.transform(X))) ** 2).sum(1)


class Cusum:
    """Two-sided CUSUM per column on standardised inputs. Rows must be in time order."""

    def __init__(self, k=0.5):
        self.k = k

    def fit(self, X):
        return self

    def score(self, X):
        hi, lo, out = np.zeros(X.shape[1]), np.zeros(X.shape[1]), np.zeros(len(X))
        for i, x in enumerate(X):
            hi = np.maximum(0, hi + x - self.k)
            lo = np.maximum(0, lo - x - self.k)
            out[i] = max(hi.max(), lo.max())
        return out


class Ewma:
    """EWMA control statistic per column on standardised inputs. Rows must be in time order."""

    def __init__(self, lam=0.2):
        self.lam = lam

    def fit(self, X):
        return self

    def score(self, X):
        z, out = np.zeros(X.shape[1]), np.zeros(len(X))
        limit = np.sqrt(self.lam / (2 - self.lam))
        for i, x in enumerate(X):
            z = self.lam * x + (1 - self.lam) * z
            out[i] = np.abs(z / limit).max()
        return out


SIMPLE = {"zscore": ZScore, "iqr": IQR, "mahalanobis": Mahalanobis, "knn": KNNDistance,
          "pca_error": PCAError, "cusum": Cusum, "ewma": Ewma}
