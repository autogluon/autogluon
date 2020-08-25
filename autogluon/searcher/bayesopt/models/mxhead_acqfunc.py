import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.ndarray import NDArray
from abc import ABC, abstractmethod
from typing import Tuple, Optional

from .nphead_acqfunc import _reshape_predictions
from ..tuning_algorithms.base_classes import SurrogateModel, AcquisitionFunction


class MXNetHeadAcquisitionFunction(AcquisitionFunction, ABC):
    """
    Base class for acquisition functions whose head are implemented in MXNet.
    Here:

        f(x, model) = h(mean, std, model.current_best())

    where h(.) is the head, and mean, std are predictive mean and stddev.

    """
    def __init__(self, model: SurrogateModel):
        super(MXNetHeadAcquisitionFunction, self).__init__(model)

    def compute_acq(self, x: np.ndarray,
                    model: Optional[SurrogateModel] = None) -> np.ndarray:
        if model is None:
            model = self.model
        predictions_list = model.predict_nd(model.convert_np_to_nd(x))
        fvals_list = []
        for mean, std in predictions_list:
            if self._head_needs_current_best():
                current_best = model.convert_np_to_nd(
                    model.current_best()).reshape((1, -1))
            else:
                current_best = None
            # Are we in batch mode? If so, mean is a matrix, whose number of columns
            # is identical to the number of samples used for fantasizing. In this
            # case, we compute the criterion values separate for every sample, and
            # average them
            do_avg = (mean.ndim == 2 and mean.shape[1] > 1)
            if do_avg:
                assert mean.shape[1] == current_best.size, \
                    "mean.shape[1] = {}, current_best.size = {} (must be the same)".format(
                        mean.shape[1], current_best.size)
                std = std.reshape((-1, 1))
            fvals = self._compute_head(mean, std, current_best)
            if do_avg:
                fvals = mx.nd.mean(fvals, axis=1)
            fvals_list.append(fvals.asnumpy().astype(x.dtype, copy=False))
        return np.mean(fvals_list, axis=0)

    def compute_acq_with_gradients(
            self, x: np.ndarray,
            model: Optional[SurrogateModel] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        if model is None:
            model = self.model
        dtype_np = x.dtype
        if x.ndim == 1:
            x = x[None, :]
        num_data = x.shape[0]

        # Loop over cases (rows of x), we need the gradients for each case
        # separately
        f_acqu = np.empty((num_data, 1), dtype=dtype_np)
        df_acqu = np.empty_like(x)
        # The current best
        if self._head_needs_current_best():
            current_best_nd = model.convert_np_to_nd(
                model.current_best()).reshape((-1,))
        else:
            current_best_nd = None

        for row in range(num_data):
            x_nd = model.convert_np_to_nd(x[row, None])
            # Record for gradient computation
            x_nd.attach_grad()
            with autograd.record():
                m_nd, s_nd = _reshape_predictions(model.predict_nd(x_nd))
                fval = mx.nd.mean(self._compute_head(
                    m_nd, s_nd, current_best_nd))
            f_acqu[row] = fval.asscalar()
            fval.backward()
            df_acqu[row] = x_nd.grad.asnumpy().astype(
                dtype_np, copy=False)
        return f_acqu, df_acqu

    @abstractmethod
    def _head_needs_current_best(self) -> bool:
        """
        :return: Is the current_best argument in _compute_head needed?
        """
        pass

    @abstractmethod
    def _compute_head(self, mean: NDArray, std: NDArray,
                      current_best: NDArray) -> NDArray:
        """
        All input and return arguments are of type mx.nd.NDArray.

        :param mean: Predictive means
        :param std: Predictive stddevs
        :param current_best: Encumbent
        :return: hvals
        """
        pass


class LCBAcquisitionFunction(MXNetHeadAcquisitionFunction):
    """
    Lower confidence bound (LCB) acquisition function:

        h(mean, std) = mean - kappa * std

    """
    def __init__(self, model: SurrogateModel, kappa: float):
        super(LCBAcquisitionFunction, self).__init__(model)
        assert kappa > 0, 'kappa must be positive'
        self.kappa = kappa

    def _head_needs_current_best(self) -> bool:
        return False

    def _compute_head(self, mean: NDArray, std: NDArray,
                      current_best: NDArray) -> NDArray:
        return mean - self.kappa * std
