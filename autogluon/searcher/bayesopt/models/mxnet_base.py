from typing import Tuple, List, Optional
import mxnet as mx
from mxnet.ndarray import NDArray
import numpy as np
import logging

from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.searcher.bayesopt.tuning_algorithms.base_classes import \
    SurrogateModel
from autogluon.searcher.bayesopt.gpmxnet.constants import DATA_TYPE
from autogluon.searcher.bayesopt.autogluon.debug_log import DebugLogPrinter

logger = logging.getLogger(__name__)


class SurrogateModelMXNet(SurrogateModel):
    """
    MXNet based model

    """
    def __init__(
            self, state: TuningJobState, active_metric: str, random_seed: int,
            ctx_nd=None, dtype_nd=None,
            debug_log: Optional[DebugLogPrinter] = None):
        if ctx_nd is None:
            ctx_nd = mx.cpu()
        if dtype_nd is None:
            dtype_nd = DATA_TYPE
        super(SurrogateModelMXNet, self).__init__(
            state, active_metric, random_seed)
        assert dtype_nd == np.float32 or dtype_nd == np.float64
        self.ctx_nd = ctx_nd
        self.dtype_nd = dtype_nd
        self._current_best = None
        self._debug_log = debug_log

    ######################################
    # shared functions for all sub-classes

    def predict(self, X: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """return prediction of X using predict_nd"""
        dtype_np, _ = self._get_dtypes(X)
        if X.ndim == 1:
            X = X[None, :]

        X_nd = self.convert_np_to_nd(X)
        predictions_nd = self.predict_nd(X_nd)

        predictions_list = [
            (m_nd.asnumpy().astype(dtype_np, copy=False),
             s_nd.asnumpy().astype(dtype_np, copy=False))
            for m_nd, s_nd in predictions_nd
        ]
        return predictions_list

    def context_for_nd(self) -> mx.Context:
        return self.ctx_nd

    def dtype_for_nd(self):
        return self.dtype_nd

    def _get_dtypes(self, x: np.ndarray):
        dtype_np = x.dtype
        if dtype_np == np.int32 or dtype_np == np.int64:
            dtype_np = np.float64
        assert dtype_np == np.float32 or dtype_np == np.float64
        dtype_nd = self.dtype_nd
        return dtype_np, dtype_nd

    def _current_best_filter_candidates(self, candidates):
        """
        Can be used by subclasses to specialize current_best.

        """
        return candidates

    def current_best(self) -> np.ndarray:
        if self._current_best is None:
            def convert(c):
                x = self.state.hp_ranges.to_ndarray(c)
                return self.convert_np_to_nd(x).reshape((1, -1))

            candidates = [
                x.candidate for x in self.state.candidate_evaluations] + \
                         self.state.pending_candidates
            if self._debug_log is not None:
                deb_msg = "[GPMXNetModel.current_best -- RECOMPUTING]\n"
                deb_msg += ("- len(candidates) = {}".format(len(candidates)))
                logger.info(deb_msg)

            if len(candidates) == 0:
                # this can happen if no evaluation is present for the current task
                # in some sense the best so far is plus infinity, any new result would be an improvement
                # In this special setting optimizing the EI is equivalent to optimizing the mean
                # and any current_best high enough will approximately give the same results
                assert len(self.state.candidate_evaluations) > 0  # other tasks will have evaluations
                # take the worst and add a few times standard dev
                values = [x.metrics[self.active_metric] for x in self.state.candidate_evaluations]
                current_worst = max(values)
                std = np.std(values)
                self._current_best = np.array([current_worst + 10 * std])
            else:
                candidates = self._current_best_filter_candidates(candidates)
                X_all = mx.nd.concat(*map(convert, candidates), dim=0)
                pred_mean = _compute_mean_across_samples(self.predict_nd(X_all))
                self._current_best = mx.nd.min(pred_mean, axis=0).asnumpy()

            logger.info(f"Current best is {self._current_best}")
        return self._current_best


def _compute_mean_across_samples(predictions_list: List[Tuple[NDArray, NDArray]]) -> NDArray:
    pred_means_list = [pred_mean for pred_mean, _ in predictions_list]
    pred_means = mx.nd.stack(*pred_means_list, axis=0)
    return mx.nd.mean(pred_means, axis=0)
