import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, NamedTuple

from ..tuning_algorithms.base_classes import SurrogateModel, AcquisitionFunction
from ..utils.density import get_quantiles


class HeadWithGradient(NamedTuple):
    hvals: np.array
    dh_dmean: np.array
    dh_dstd: np.array


class MeanStdAcquisitionFunction(AcquisitionFunction, ABC):
    """
    Base class for standard acquisition functions which depend on predictive
    mean and stddev. Subclasses have to implement the head and its derivatives
    w.r.t. mean and std:

        f(x, model) = h(mean, std, model.current_best())

    NOTE that acquisition functions will always be *minimized*!

    """
    def __init__(self, model: SurrogateModel):
        super(MeanStdAcquisitionFunction, self).__init__(model)
        assert 'mean' in model.keys_predict()
        assert 'std' in model.keys_predict()

    def compute_acq(self, inputs: np.ndarray,
                    model: Optional[SurrogateModel] = None) -> np.ndarray:
        if model is None:
            model = self.model
        assert 'mean' in model.keys_predict()
        assert 'std' in model.keys_predict()

        if inputs.ndim == 1:
            inputs = inputs.reshape((1, -1))
        predictions_list = model.predict(inputs)
        if self._head_needs_current_best():
            current_best_list = model.current_best()
        else:
            current_best_list = [None] * len(predictions_list)
        fvals_list = []
        # Loop over MCMC samples (if any)
        for predictions, current_best in zip(predictions_list, current_best_list):
            means = predictions['mean']
            stds = predictions['std']
            num_fantasies = means.shape[1] if means.ndim == 2 else 1
            if num_fantasies > 1:
                stds = stds.reshape((-1, 1))
                if current_best is not None:
                    assert current_best.size == num_fantasies, \
                        "mean.shape[1] = {}, current_best.size = {} (must be the same)".format(
                            num_fantasies, current_best.size)
                    current_best = current_best.reshape((1, -1))
            else:
                means = means.reshape((-1,))
                stds = stds.reshape((-1,))
                if current_best is not None:
                    current_best = current_best.reshape((1,))
            fvals = self._compute_heads(means, stds, current_best)
            # Average over fantasies if there are any
            if num_fantasies > 1:
                fvals = np.mean(fvals, axis=1)
            fvals_list.append(fvals)

        return np.mean(fvals_list, axis=0)

    def compute_acq_with_gradient(
            self, input: np.ndarray,
            model: Optional[SurrogateModel] = None) -> \
            Tuple[float, np.ndarray]:
        if model is None:
            model = self.model
        assert 'mean' in model.keys_predict()
        assert 'std' in model.keys_predict()

        predictions_list = model.predict(input.reshape(1, -1))
        if self._head_needs_current_best():
            current_best_list = model.current_best()
        else:
            current_best_list = [None] * len(predictions_list)
        fvals = []
        head_gradients = []

        for predictions, current_best in zip(predictions_list, current_best_list):
            mean = predictions['mean'].reshape((-1,))
            num_fantasies = mean.size
            std = predictions['std'].reshape((1,))
            if current_best is not None:
                assert current_best.size == num_fantasies
                current_best = current_best.reshape((-1,))
            head_result = self._compute_head_and_gradient(mean, std, current_best)
            fvals.append(np.mean(head_result.hvals))
            # Compose head_gradient for backward_gradient call. This involves
            # reshaping. Also, we are averaging over fantasies (if
            # num_fantasies > 1), which is not done in
            # _compute_head_and_gradient (dh_dmean, dh_dstd have the same shape
            # as mean), so have to divide by num_fantasies
            head_gradients.append({
                'mean': head_result.dh_dmean.reshape(
                    predictions['mean'].shape) / num_fantasies,
                'std': np.array([np.mean(head_result.dh_dstd)]).reshape(
                    predictions['std'].shape)})

        # Gradients are computed by the model
        gradient_list = model.backward_gradient(input, head_gradients)
        # Average over MCMC samples (if any)
        fval = np.mean(fvals)
        gradient = np.mean(gradient_list, axis=0)
        return fval, gradient

    @abstractmethod
    def _head_needs_current_best(self) -> bool:
        """
        :return: Is the current_best argument in _compute_head needed?
        """
        pass

    @abstractmethod
    def _compute_heads(
            self, means: np.ndarray, stds: np.ndarray,
            current_best: Optional[np.ndarray]) -> np.ndarray:
        """
        If mean has >1 columns, both std and current_best are supposed to be
        broadcasted. The return value has the same shape as mean.

        :param means: Predictive means, shape (n, nf) or (n,)
        :param stds: Predictive stddevs, shape (n, 1) or (n,)
        :param current_best: Incumbent, shape (1, nf) or (1,)
        :return: h(means, stds, current_best), same shape as means
        """
        pass

    @abstractmethod
    def _compute_head_and_gradient(
            self, mean: np.ndarray, std: np.ndarray,
            current_best: Optional[np.ndarray]) -> HeadWithGradient:
        """
        Computes both head value and head gradients, for a single input.

        :param mean: Predictive mean, shape (nf,)
        :param std: Predictive stddev, shape (1,)
        :param current_best: Incumbent, shape (nf,)
        :return: All HeadWithGradient fields have same shape as mean
        """
        pass


class EIAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Minus expected improvement acquisition function
    (minus because the convention is to always minimize acquisition functions)

    """
    def __init__(self, model: SurrogateModel, jitter: float = 0.01):
        super().__init__(model)
        self.jitter = jitter

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_heads(
            self, means: np.ndarray, stds: np.ndarray,
            current_best: Optional[np.ndarray]) -> np.ndarray:
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        return (-stds) * (u * Phi + phi)

    def _compute_head_and_gradient(
            self, mean: np.ndarray, std: np.ndarray,
            current_best: Optional[np.ndarray]) -> HeadWithGradient:
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        f_acqu = std * (u * Phi + phi)
        return HeadWithGradient(
            hvals=-f_acqu,
            dh_dmean=Phi,
            dh_dstd=-phi)


class LCBAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Lower confidence bound (LCB) acquisition function:

        h(mean, std) = mean - kappa * std

    """
    def __init__(self, model: SurrogateModel, kappa: float):
        super().__init__(model)
        assert kappa > 0, 'kappa must be positive'
        self.kappa = kappa

    def _head_needs_current_best(self) -> bool:
        return False

    def _compute_heads(
            self, means: np.ndarray, stds: np.ndarray,
            current_best: Optional[np.ndarray]) -> np.ndarray:
        return means - stds * self.kappa

    def _compute_head_and_gradient(
            self, mean: np.ndarray, std: np.ndarray,
            current_best: Optional[np.ndarray]) -> HeadWithGradient:
        ones_like_mean = np.ones_like(mean)
        return HeadWithGradient(
            hvals=mean - std * self.kappa,
            dh_dmean=ones_like_mean,
            dh_dstd=(-self.kappa) * ones_like_mean)
