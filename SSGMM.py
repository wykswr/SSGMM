import numpy as np
from scipy.stats import multivariate_normal


class SSGaussianMixture:
    def __init__(self, n_categories):
        self.n_features = None
        self.n_categories = n_categories
        self.mus = None
        self.sigmas = None
        self.pis = np.array([1 / n_categories] * n_categories)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, max_iter=100):
        self.n_features = X_train.shape[1]
        self.mus = np.array([np.random.randn(self.n_features)] * self.n_categories)
        self.sigmas = [np.eye(self.n_features)] * self.n_categories
        z_mat = np.eye(self.n_categories)[y_train]
        for i in range(max_iter):
            # M step
            gammas = self.cal_gamma(X_test)
            n_vec = np.array([z_mat[:, k].sum() + gammas[:, k].sum() for k in range(self.n_categories)])
            # E step
            for k in range(self.n_categories):
                # update mu
                self.mus[k] = (z_mat[:, k] @ X_train + gammas[:, k] @ X_test) / n_vec[k]
                # update sigma
                new_sigma = np.zeros([self.n_features, self.n_features])
                for idx, x in enumerate(X_train):
                    delta = np.array([x - self.mus[k]])
                    new_sigma += (z_mat[idx, k] * delta.T @ delta)
                for idx, x in enumerate(X_test):
                    delta = np.array([x - self.mus[k]])
                    new_sigma += (gammas[idx, k] * delta.T @ delta)
                self.sigmas[k] = new_sigma / n_vec[k]
                # update pi
                self.pis[k] = (z_mat[:, k].sum() + gammas[:, k].sum()) / n_vec.sum()

    def __gamma(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: vector
        :return: posterior probability of all latent components
        """
        probs = np.array(
            [self.pis[k] * multivariate_normal.pdf(x, mean=self.mus[k], cov=self.sigmas[k]) for k in
             range(self.n_categories)])
        return probs / probs.sum()

    def cal_gamma(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.__gamma(x) for x in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.cal_gamma(X).max(axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        :param X: matrix
        :return: most likely labels
        """
        gammas = self.cal_gamma(X)
        return np.array([np.argmax(probs) for probs in gammas])
