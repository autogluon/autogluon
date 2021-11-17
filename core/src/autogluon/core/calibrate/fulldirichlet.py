import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss

from .multinomial import MultinomialRegression
from .utils import clip_for_log


class FullDirichletCalibrator(BaseEstimator, RegressorMixin):
    def __init__(self, reg_lambda_list=0.0, reg_mu=None, weights_init=None,
                 initializer='identity', reg_norm=False, ref_row=True,
                 optimizer='auto'):
        """
        Params:
            weights_init: (nd.array) weights used for initialisation, if None
            then idendity matrix used. Shape = (n_classes - 1, n_classes + 1)
            optimizer: string ('auto', 'newton', 'fmin_l_bfgs_b')
                If 'auto': then 'newton' for less than 37 classes and
                fmin_l_bfgs_b otherwise
                If 'newton' then uses our implementation of a Newton method
                If 'fmin_l_bfgs_b' then uses scipy.ptimize.fmin_l_bfgs_b which
                implements a quasi Newton method
        """
        self.reg_lambda = reg_lambda_list
        self.reg_mu = reg_mu  # Complementary L2 regularization. (Off-diagonal regularization)
        self.weights_init = weights_init  # Input weights for initialisation
        self.initializer = initializer
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.weights_ = self.weights_init

        k = np.shape(X)[1]

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        _X = np.copy(X)
        _X = np.log(clip_for_log(_X))
        _X_val = np.copy(X_val)
        _X_val = np.log(clip_for_log(X_val))

        self.calibrator_ = MultinomialRegression(method='Full',
                                                 reg_lambda=self.reg_lambda,
                                                 reg_mu=self.reg_mu,
                                                 reg_norm=self.reg_norm,
                                                 ref_row=self.ref_row,
                                                 optimizer=self.optimizer)
        self.calibrator_.fit(_X, y, *args, **kwargs)
        final_loss = log_loss(y_val, self.calibrator_.predict_proba(_X_val))

        return self

    @property
    def weights(self):
        if self.calibrator_ is not None:
            return self.calibrator_.weights_
        return self.weights_init

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict_proba(S))

    def predict(self, S):
        S = np.log(clip_for_log(S))
        return np.asarray(self.calibrator_.predict(S))
