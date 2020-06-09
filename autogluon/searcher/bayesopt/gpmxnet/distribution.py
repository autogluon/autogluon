from abc import ABC, abstractmethod
import numpy as np
from scipy.special import gammaln
import numbers

from autogluon.searcher.bayesopt.gpmxnet.constants import MIN_POSTERIOR_VARIANCE

__all__ = ['Distribution',
           'Gamma',
           'Uniform',
           'Normal',
           'LogNormal',
           'Horseshoe']


class Distribution(ABC):
    @abstractmethod
    def negative_log_density(self, F, x):
        """
        Negative log density, computed in MXNet. lower and upper limits are
        ignored. If x is not a scalar, the distribution is i.i.d. over all
        entries.
        """
        pass


class Gamma(Distribution):
    """
    Gamma(mean, alpha):

        p(x) = C(alpha, beta) x^{alpha - 1} exp( -beta x), beta = alpha / mean,
        C(alpha, beta) = beta^alpha / Gamma(alpha)

    """
    def __init__(self, mean, alpha):
        self._assert_positive_number(mean, 'mean')
        self._assert_positive_number(alpha, 'alpha')
        self.mean = np.maximum(mean, MIN_POSTERIOR_VARIANCE)
        self.alpha = np.maximum(alpha, MIN_POSTERIOR_VARIANCE)
        self.beta = self.alpha / self.mean
        self.log_const = gammaln(self.alpha) - self.alpha * np.log(self.beta)
        self.__call__ = self.negative_log_density

    @staticmethod
    def _assert_positive_number(x, name):
        assert isinstance(x, numbers.Real) and x > 0.0, \
            "{} = {}, must be positive number".format(name, x)

    def negative_log_density(self, F, x):
        x_safe = F.maximum(x, MIN_POSTERIOR_VARIANCE)
        return F.sum(
            (1.0 - self.alpha) * F.log(x_safe) + self.beta * x_safe +
            self.log_const)

    def __call__(self, F, x):
        return self.negative_log_density(F, x)


class Uniform(Distribution):
    def __init__(self, lower: float, upper: float):
        self.log_const = np.log(upper - lower)
        self.__call__ = self.negative_log_density

    def negative_log_density(self, F, x):
        return F.sum(F.ones_like(x)) * self.log_const

    def __call__(self, F, x):
        return self.negative_log_density(F, x)


class Normal(Distribution):
    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma
        self.__call__ = self.negative_log_density

    def negative_log_density(self, F, x):
        return F.sum(F.square(x - self.mean)) * (0.5 / np.square(self.sigma))

    def __call__(self, F, x):
        return self.negative_log_density(F, x)


class LogNormal(Distribution):
    def __init__(self, mean: float, sigma: float):
        self.mean = mean
        self.sigma = sigma
        self.__call__ = self.negative_log_density

    def negative_log_density(self, F, x):
        x_safe = F.maximum(x, MIN_POSTERIOR_VARIANCE)
        return F.sum(
            F.log(x_safe * self.sigma) +
            F.square(F.log(x_safe) - self.mean) * (0.5 / np.square(self.sigma)))

    def __call__(self, F, x):
        return self.negative_log_density(F, x)


class Horseshoe(Distribution):
    def __init__(self, s: float):
        assert s > 0.0
        self.s = max(s, MIN_POSTERIOR_VARIANCE)
        self.__call__ = self.negative_log_density

    def negative_log_density(self, F, x):
        arg = F.maximum(3.0 * F.square(self.s / x), MIN_POSTERIOR_VARIANCE)
        return -F.sum(F.log(F.log1p(arg)))

    def __call__(self, F, x):
        return self.negative_log_density(F, x)
