import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import log_loss

from .multinomial import MultinomialRegression
from .utils import clip_for_log


class MatrixScaling(BaseEstimator, RegressorMixin):
    def __init__(self, reg_lambda_list=[0.0], reg_mu_list=[None],
                 logit_input=False, logit_constant=None,
                 weights_init=None, initializer='identity'):
        self.weights_init = weights_init
        self.logit_input = logit_input
        self.logit_constant = logit_constant
        self.reg_lambda_list = reg_lambda_list
        self.reg_mu_list = reg_mu_list
        self.initializer = initializer

    def __setup(self):
        self.reg_lambda = 0.0
        self.reg_mu = None
        self.calibrator_ = None
        self.weights_ = self.weights_init

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):

        self.__setup()

        k = np.shape(X)[1]

        if X_val is None:
            X_val = X.copy()
            y_val = y.copy()

        if self.logit_input == False:
            _X = np.copy(X)
            _X = np.log(clip_for_log(_X))
            _X_val = np.copy(X_val)
            _X_val = np.log(clip_for_log(X_val))
            if self.logit_constant is None:
                _X = _X - _X[:, -1].reshape(-1, 1).repeat(k, axis=1)
                _X_val = _X_val[:, -1].reshape(-1, 1).repeat(k, axis=1)
            else:
                _X = _X - self.logit_constant
                _X_val = _X_val - self.logit_constant
        else:
            _X = np.copy(X)
            _X_val = np.copy(X_val)

        for i in range(0, len(self.reg_lambda_list)):
            for j in range(0, len(self.reg_mu_list)):
                tmp_cal = MultinomialRegression(method='Full',
                                                reg_lambda=self.reg_lambda_list[i],
                                                reg_mu=self.reg_mu_list[j])
                tmp_cal.fit(_X, y, *args, **kwargs)
                tmp_loss = log_loss(y_val, tmp_cal.predict_proba(_X_val))

                if (i + j) == 0:
                    final_cal = tmp_cal
                    final_loss = tmp_loss
                    final_reg_lambda = self.reg_lambda_list[i]
                    final_reg_mu = self.reg_mu_list[j]
                elif tmp_loss < final_loss:
                    final_cal = tmp_cal
                    final_loss = tmp_loss
                    final_reg_lambda = self.reg_lambda_list[i]
                    final_reg_mu = self.reg_mu_list[j]

        self.calibrator_ = final_cal
        self.reg_lambda = final_reg_lambda
        self.reg_mu = final_reg_mu
        self.weights_ = self.calibrator_.weights_

        return self

    @property
    def coef_(self):
        return self.calibrator_.coef_

    @property
    def intercept_(self):
        return self.calibrator_.intercept_

    def predict_proba(self, S):
        k = np.shape(S)[1]

        if self.logit_input == False:
            _S = np.log(clip_for_log(np.copy(S)))
            if self.logit_constant is None:
                _S = _S - _S[:, -1].reshape(-1, 1).repeat(k, axis=1)
            else:
                _S = _S - self.logit_constant
        else:
            _S = np.copy(S)

        return np.asarray(self.calibrator_.predict_proba(_S))

    def predict(self, S):
        k = np.shape(S)[1]

        if self.logit_input == False:
            _S = np.log(clip_for_log(np.copy(S)))
            if self.logit_constant is None:
                _S = _S - _S[:, -1].reshape(-1, 1).repeat(k, axis=1)
            else:
                _S = _S - self.logit_constant
        else:
            _S = np.copy(S)

        return np.asarray(self.calibrator_.predict(_S))
