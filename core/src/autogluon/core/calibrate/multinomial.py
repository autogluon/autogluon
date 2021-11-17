from __future__ import division

import logging

import jax
import jax.numpy as np
import numpy as raw_np
import scipy
import scipy.linalg
import scipy.optimize
from jax.config import config
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import label_binarize

from .utils import clip_jax

config.update("jax_enable_x64", True)


class MultinomialRegression(BaseEstimator, RegressorMixin):
    def __init__(self, weights_0=None, method='Full', initializer='identity',
                 reg_format=None, reg_lambda=0.0, reg_mu=None, reg_norm=False,
                 ref_row=True, optimizer='auto'):
        """
        Params:
            optimizer: string ('auto', 'newton', 'fmin_l_bfgs_b')
                If 'auto': then 'newton' for less than 37 classes and
                fmin_l_bfgs_b otherwise
                If 'newton' then uses our implementation of a Newton method
                If 'fmin_l_bfgs_b' then uses scipy.ptimize.fmin_l_bfgs_b which
                implements a quasi Newton method
        """
        if method not in ['Full', 'Diag', 'FixDiag']:
            raise (ValueError('method {} not avaliable'.format(method)))

        self.weights_0 = weights_0
        self.method = method
        self.initializer = initializer
        self.reg_format = reg_format
        self.reg_lambda = reg_lambda
        self.reg_mu = reg_mu  # If number, then ODIR is applied
        self.reg_norm = reg_norm
        self.ref_row = ref_row
        self.optimizer = optimizer

    def __setup(self):
        self.classes = None
        self.weights_ = self.weights_0
        self.weights_0_ = self.weights_0

    @property
    def coef_(self):
        return self.weights_[:, :-1]

    @property
    def intercept_(self):
        return self.weights_[:, -1]

    def predict_proba(self, S):

        S_ = np.hstack((S, np.ones((len(S), 1))))

        return np.asarray(_calculate_outputs(self.weights_, S_))

    # FIXME Should we change predict for the argmax?
    def predict(self, S):

        return np.asarray(self.predict_proba(S))

    def fit(self, X, y, *args, **kwargs):

        self.__setup()

        X_ = np.hstack((X, np.ones((len(X), 1))))

        self.classes = raw_np.unique(y)

        k = len(self.classes)

        if self.reg_norm:
            if self.reg_mu is None:
                self.reg_lambda = self.reg_lambda / (k * (k + 1))
            else:
                self.reg_lambda = self.reg_lambda / (k * (k - 1))
                self.reg_mu = self.reg_mu / k

        target = label_binarize(y=y, classes=self.classes)

        if k == 2:
            target = np.hstack([1 - target, target])

        n, m = X_.shape

        XXT = (X_.repeat(m, axis=1) * np.hstack([X_] * m)).reshape((n, m, m))

        logging.debug(self.method)

        self.weights_0_ = self._get_initial_weights(self.initializer)

        if (self.optimizer == 'newton'
                or (self.optimizer == 'auto' and k <= 36)):
            weights = _newton_update(self.weights_0_, X_, XXT, target, k,
                                     self.method, reg_lambda=self.reg_lambda,
                                     reg_mu=self.reg_mu, ref_row=self.ref_row,
                                     initializer=self.initializer,
                                     reg_format=self.reg_format)
        elif (self.optimizer == 'fmin_l_bfgs_b'
              or (self.optimizer == 'auto' and k > 36)):

            _gradient_np = lambda *args, **kwargs: raw_np.array(_gradient(*args, **kwargs))

            res = scipy.optimize.fmin_l_bfgs_b(func=_objective,
                                               fprime=_gradient_np,
                                               x0=self.weights_0_,
                                               args=(X_, XXT, target, k,
                                                     self.method,
                                                     self.reg_lambda,
                                                     self.reg_mu, self.ref_row,
                                                     self.initializer,
                                                     self.reg_format),
                                               maxls=128,
                                               factr=1.0)
            weights = res[0]
        else:
            raise (ValueError('Unknown optimizer: {}'.format(self.optimizer)))

        self.weights_ = _get_weights(weights, k, self.ref_row, self.method)

        return self

    def _get_initial_weights(self, ref_row, initializer='identity'):
        ''' Returns an array containing only the weights of the full weight
        matrix.

        '''

        if initializer not in ['identity', None]:
            raise ValueError

        k = len(self.classes)

        if self.weights_0_ is None:
            if initializer == 'identity':
                weights_0 = _get_identity_weights(k, ref_row, self.method)
            else:
                if self.method == 'Full':
                    weights_0 = np.zeros(k * (k + 1))
                elif self.method == 'Diag':
                    weights_0 = np.zeros(2 * k)
                elif self.method == 'FixDiag':
                    weights_0 = np.zeros(1)
        else:
            weights_0 = self.weights_0_

        return weights_0


def _objective(params, *args):
    (X, _, y, k, method, reg_lambda, reg_mu, ref_row, _, reg_format) = args
    weights = _get_weights(params, k, ref_row, method)
    outputs = clip_jax(_calculate_outputs(weights, X))
    loss = np.mean(-np.log(np.sum(y * outputs, axis=1)))

    if reg_mu is None:
        if reg_format == 'identity':
            reg = np.hstack([np.eye(k), np.zeros((k, 1))])
        else:
            reg = np.zeros((k, k + 1))
        loss = loss + reg_lambda * np.sum((weights - reg) ** 2)
    else:
        weights_hat = weights - np.hstack([weights[:, :-1] * np.eye(k),
                                           np.zeros((k, 1))])
        loss = loss + reg_lambda * np.sum(weights_hat[:, :-1] ** 2) + \
               reg_mu * np.sum(weights_hat[:, -1] ** 2)

    return loss


_gradient = jax.grad(_objective, argnums=0)

_hessian = jax.hessian(_objective, argnums=0)


def _get_weights(params, k, ref_row, method):
    ''' Reshapes the given params (weights) into the full matrix including 0
    '''

    if method in ['Full', None]:
        raw_weights = params.reshape(-1, k + 1)
        # weights = np.zeros([k, k+1])
        # weights[:-1, :] = params.reshape(-1, k + 1)

    elif method == 'Diag':
        raw_weights = np.hstack([np.diag(params[:k]),
                                 params[k:].reshape(-1, 1)])
        # weights[:, :-1][np.diag_indices(k)] = params[:]

    elif method == 'FixDiag':
        raw_weights = np.hstack([np.eye(k) * params[0], np.zeros((k, 1))])
        # weights[np.dgag_indices(k - 1)] = params[0]
        # weights[np.diag_indices(k)] = params[0]
    else:
        raise (ValueError("Unknown calibration method {}".format(method)))

    if ref_row:
        weights = raw_weights - np.repeat(
            raw_weights[-1, :].reshape(1, -1), k, axis=0)
    else:
        weights = raw_weights

    return weights


def _get_identity_weights(n_classes, ref_row, method):
    raw_weights = None

    if (method is None) or (method == 'Full'):
        raw_weights = np.zeros((n_classes, n_classes + 1)) + \
                      np.hstack([np.eye(n_classes), np.zeros((n_classes, 1))])
        raw_weights = raw_weights.ravel()

    elif method == 'Diag':
        raw_weights = np.hstack([np.ones(n_classes), np.zeros(n_classes)])

    elif method == 'FixDiag':
        raw_weights = np.ones(1)

    return raw_weights.ravel()


def _calculate_outputs(weights, X):
    mul = np.dot(X, weights.transpose())
    return _softmax(mul)


def _softmax(X):
    """Compute the softmax of matrix X in a numerically stable way."""
    shiftx = X - np.max(X, axis=1).reshape(-1, 1)
    exps = np.exp(shiftx)
    return exps / np.sum(exps, axis=1).reshape(-1, 1)


def _newton_update(weights_0, X, XX_T, target, k, method_, maxiter=int(1024),
                   ftol=1e-12, gtol=1e-8, reg_lambda=0.0, reg_mu=None,
                   ref_row=True, initializer=None, reg_format=None):
    L_list = [float(_objective(weights_0, X, XX_T, target, k, method_,
                               reg_lambda, reg_mu, ref_row, initializer,
                               reg_format))]

    weights = weights_0.copy()

    # TODO move this to the initialization
    if method_ is None:
        weights = np.zeros_like(weights)

    for i in range(0, maxiter):

        gradient = _gradient(weights, X, XX_T, target, k, method_, reg_lambda,
                             reg_mu, ref_row, initializer, reg_format)

        if np.abs(gradient).sum() < gtol:
            break

        # FIXME hessian is ocasionally NaN
        hessian = _hessian(weights, X, XX_T, target, k, method_, reg_lambda,
                           reg_mu, ref_row, initializer, reg_format)

        if method_ == 'FixDiag':
            updates = gradient / hessian
        else:
            try:
                inverse = scipy.linalg.pinv2(hessian)
                updates = np.matmul(inverse, gradient)
            except (raw_np.linalg.LinAlgError, ValueError) as err:
                logging.error(err)
                updates = gradient

        for step_size in np.hstack((np.linspace(1, 0.1, 10),
                                    np.logspace(-2, -32, 31))):

            tmp_w = weights - (updates * step_size).ravel()

            if np.any(np.isnan(tmp_w)):
                logging.debug("{}: There are NaNs in tmp_w".format(method_))

            L = _objective(tmp_w, X, XX_T, target, k, method_, reg_lambda,
                           reg_mu, ref_row, initializer, reg_format)

            if (L - L_list[-1]) < 0:
                break

        L_list.append(float(L))

        logging.debug("{}: after {} iterations log-loss = {:.7e}, sum_grad = {:.7e}".format(
            method_, i, L, np.abs(gradient).sum()))

        if np.isnan(L):
            logging.error("{}: log-loss is NaN".format(method_))
            break

        if i >= 5:
            if (float(raw_np.min(raw_np.diff(L_list[-5:]))) > -ftol) & \
                    (float(raw_np.sum(raw_np.diff(L_list[-5:])) > 0) == 0):
                weights = tmp_w.copy()
                logging.debug('{}: Terminate as there is not enough changes on loss.'.format(
                    method_))
                break

        if (L_list[-1] - L_list[-2]) > 0:
            logging.debug('{}: Terminate as the loss increased {}.'.format(
                method_, np.diff(L_list[-2:])))
            break
        else:
            weights = tmp_w.copy()

    L = _objective(weights, X, XX_T, target, k, method_,
                   reg_lambda, reg_mu, ref_row, initializer, reg_format)

    logging.debug("{}: after {} iterations final log-loss = {:.7e}, sum_grad = {:.7e}".format(
        method_, i, L, np.abs(gradient).sum()))

    return weights
