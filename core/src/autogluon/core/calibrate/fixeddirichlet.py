import autograd
import autograd.numpy as np

from sklearn.base import BaseEstimator, RegressorMixin

def clip_for_log(X):
    eps = np.finfo(X.dtype).tiny
    return np.clip(X, eps, 1-eps)

class FixedDiagonalDirichletCalibrator(BaseEstimator, RegressorMixin):
    def fit(self, X, y, batch_size=128, lr=1e-3, beta_1=0.9, beta_2=0.999,
            eeps=1e-8, maxiter=int(1024), factor=1e-4, *args, **kwargs):
        eps = np.finfo(X.dtype).min
        X_ = np.log(clip_for_log(X))
        k = np.shape(X_)[1]
        n_y = np.shape(X_)[0]
        y_hot = np.zeros((len(y), k))
        for i in range(0, k):
            y_hot[y==i, i] = 1.0
        L = []
        m = np.zeros(1)
        v = np.zeros(1)
        T = 1.0
        fin_T = T
        if batch_size is None:
            batch_size = n_y

        get_gradient = autograd.grad(self._objective, 0)

        batch_idx = np.hstack([np.arange(0, n_y, batch_size), n_y])

        batch_num = len(batch_idx) - 1

        for i in range(0, maxiter):

            batch_L = []

            for j in range(0, batch_num):

                L_t = self._objective(T, X, y_hot)

                g_t = get_gradient(T, X, y_hot)

                m = beta_1 * m + (1 - beta_1) * g_t

                v = beta_2 * v + (1 - beta_2) * g_t * g_t

                T = T - lr * m / (v ** 0.5 + eeps)

                batch_L.append(L_t)

                # print('Iteration: ' + str(i) + ', Batch: ' + str(j) + ', Theta: ' + str(theta[:8]))

            L.append(np.sum(batch_L))

            per_idx = np.random.permutation(n_y)

            X = X[per_idx]

            y_hot = y_hot[per_idx]

            y = y[per_idx]

            if len(L) >= 2:
                if L[-1] < np.min(L[:-1]):
                    fin_T = T.copy()

            if len(L) > 32:

                previous_opt = np.min(L.copy()[:-32])

                current_opt = np.min(L.copy()[-32:])

                if previous_opt - current_opt <= np.abs(previous_opt * factor):
                    break

        self.weights_ = fin_T

        return self

    def _objective(self, T, X, y_hot):

        tmp_prod = T * np.log(X)

        prob_y = np.exp(tmp_prod) / np.sum(np.exp(tmp_prod), axis=1).reshape(-1, 1)

        loss = -np.sum(np.log(np.sum(prob_y * y_hot, axis=1)))

        return loss

    @property
    def coef_(self):
        return self.weights_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        k = np.shape(S)[1]
        tmp_prod = self.weights_ * S
        prob_y = np.exp(tmp_prod) / np.sum(np.exp(tmp_prod), axis=1).reshape(-1, 1)
        return prob_y

    def predict(self, S):
        S = np.log(clip_for_log(S))
        k = np.shape(S)[1]
        tmp_prod = self.weights_ * S
        prob_y = np.exp(tmp_prod) / np.sum(np.exp(tmp_prod), axis=1).reshape(-1, 1)
        return prob_y


if __name__ == '__main__':
    from sklearn import datasets

    # import some data to play with
    iris = datasets.load_iris()
    X = iris.data[:, :3]  # we only take the first two features.
    y = iris.target

    softmax = lambda z:np.divide(np.exp(z).T, np.sum(np.exp(z), axis=1)).T
    S = softmax(X)

    print(S)
    print(y)
    calibrator = FixedDiagonalDirichletCalibrator()
    calibrator.fit(S, y)
    print(calibrator.predict_proba(S))
