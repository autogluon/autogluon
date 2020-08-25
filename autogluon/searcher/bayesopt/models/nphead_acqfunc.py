import numpy as np
import mxnet as mx
from mxnet import autograd
from mxnet.ndarray import NDArray
from abc import ABC, abstractmethod
from typing import Tuple, List, NamedTuple, Optional

from ..tuning_algorithms.base_classes import SurrogateModel, AcquisitionFunction
from ..utils.density import get_quantiles


class NumPyHeadResult(NamedTuple):
    hvals: np.array
    dh_dmean: np.array
    dh_dstd: np.array


class NumPyHeadAcquisitionFunction(AcquisitionFunction, ABC):
    """
    Base class for acquisition functions whose head are implemented in NumPy
    (and not in MXNet). Here:

        f(x, model) = h(mean, std, model.current_best())

    where h(.) is the head, mean, std predictive mean and stddev.
    If h(.) can be implemented in MXNet, it is more efficient to do that, but
    often MXNet lacks functionality.

    NOTE that acquisition functions will always be *minimized*!

    """
    def __init__(self, model: SurrogateModel):
        super(NumPyHeadAcquisitionFunction, self).__init__(model)

    def compute_acq(self, x: np.ndarray,
                    model: Optional[SurrogateModel] = None) -> np.ndarray:
        if model is None:
            model = self.model
        predictions_list = model.predict(x)
        fvals_list = []
        for mean, std in predictions_list:
            if self._head_needs_current_best():
                current_best = model.current_best().reshape((1, -1))
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
            fvals = self._compute_head(mean, std, current_best).hvals
            if do_avg:
                fvals = np.mean(fvals, axis=1)
            fvals_list.append(fvals)
        return np.mean(fvals_list, axis=0)

    def compute_acq_with_gradients(
            self, x: np.ndarray,
            model: Optional[SurrogateModel] = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        if model is None:
            model = self.model
        dtype_nd = model.dtype_for_nd()
        dtype_np = x.dtype
        ctx = model.context_for_nd()
        if x.ndim == 1:
            x = x[None, :]
        num_data = x.shape[0]

        # Loop over cases (rows of x), we need the gradients for each case
        # separately
        f_acqu = np.empty((num_data, 1), dtype=dtype_np)
        df_acqu = np.empty_like(x)
        # The current best
        if self._head_needs_current_best():
            current_best = model.current_best().reshape((-1,))
        else:
            current_best = None

        dfdm_nd, dfds_nd, num_samples = None, None, None
        for row in range(num_data):
            x_nd = model.convert_np_to_nd(x[row, None])
            # Compute heads m_nd, s_nd while recording
            x_nd.attach_grad()
            with autograd.record():
                m_nd, s_nd = _reshape_predictions(model.predict_nd(x_nd))
                if dtype_np != dtype_nd:
                    m_nd = m_nd.astype(dtype_np)
                    s_nd = s_nd.astype(dtype_np)

            # Compute head gradients in NumPy
            head_result = self._compute_head(
                m_nd.asnumpy(), s_nd.asnumpy(), current_best)
            f_acqu[row] = np.mean(head_result.hvals)
            if row == 0:
                num_samples = m_nd.size
                dfdm_nd = mx.nd.array(head_result.dh_dmean, ctx=ctx, dtype=dtype_np)
                dfds_nd = mx.nd.array(head_result.dh_dstd, ctx=ctx, dtype=dtype_np)
            else:
                dfdm_nd[:] = head_result.dh_dmean
                dfds_nd[:] = head_result.dh_dstd

            # Backward with specific head gradients
            autograd.backward([m_nd, s_nd], [dfdm_nd, dfds_nd])
            df_acqu[row] = x_nd.grad.asnumpy().astype(
                dtype_np, copy=False) / num_samples
        return f_acqu, df_acqu

    @abstractmethod
    def _head_needs_current_best(self) -> bool:
        """
        :return: Is the current_best argument in _compute_head needed?
        """
        pass

    @abstractmethod
    def _compute_head(
            self, mean: np.ndarray, std: np.ndarray,
            current_best: Optional[np.ndarray]) -> NumPyHeadResult:
        """
        Note that the head value will always be *minimized*, that is why we for example return
        minus the expected improvement

        :param mean: Predictive means
        :param std: Predictive stddevs
        :param current_best: Encumbent
        :return: NumPyHeadResult containing hvals, dh_dmean, dh_dstd
        """
        pass


def _reshape_predictions(predictions: List[Tuple[NDArray, NDArray]]) \
        -> Tuple[NDArray, NDArray]:
    """

    :param predictions: prediction of one sample. It's list of tuples, which are predicted mean
    and variance. The predicted means can have more than 1 column in the case of fantasizing

    predicted mean can be mx.nd.array([1,]) or mx.nd.array([1,2,3]) (fantasizing)
    predicted variance is always mx.nd.array([1,])

    An example, 3 MCMC samples with 2 fantasized output
    predictions = [(mx.nd.array([1,2]), mx.nd.array([0.1,])),
                   (mx.nd.array([2,3]), mx.nd.array([0.2,])),
                   (mx.nd.array([3,4]), mx.nd.array([0.3,])]

    The method returns (mx.nd.array([1,2,2,3,3,4]), mx.nd.array([0.1,0.1,0.2,0.2,0.3,0.3]))

    :return: a tuple of two mx.nd.NDArray, which are flattened mean and variances
    """
    m_nd_list, s_nd_list = zip(*predictions)
    n_samples = len(predictions)
    n_fantasizing = m_nd_list[0].size

    m_nd = mx.nd.concat(*m_nd_list, dim=0)
    s_nd = mx.nd.concat(*s_nd_list, dim=0)
    m_nd = m_nd.reshape(n_samples, n_fantasizing)
    s_nd = s_nd.reshape(n_samples, 1)

    s_nd = mx.nd.broadcast_axis(s_nd, axis=1, size=n_fantasizing)
    return m_nd.reshape((-1)), s_nd.reshape((-1))


class EIAcquisitionFunction(NumPyHeadAcquisitionFunction):
    """
    Minus expected improvement acquisition function
    (minus because the convention is to always minimize acquisition functions)

    """
    def __init__(self, model: SurrogateModel, jitter: float = 0.01):
        super().__init__(model)
        self.jitter = jitter

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_head(self, mean: np.array, std: np.array, current_best: Optional[np.array]):
        """
        Returns -1 times the expected improvement and gradients with respect to mean
        and standard deviation
        """
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        f_acqu = std * (u * Phi + phi)
        return NumPyHeadResult(
            hvals=-f_acqu,
            dh_dmean=Phi,
            dh_dstd=-phi)
