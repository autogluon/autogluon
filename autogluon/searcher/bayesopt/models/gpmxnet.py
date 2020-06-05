from typing import Tuple, NamedTuple, Dict, Union, List, Optional
import mxnet as mx
import numpy as np
import logging

from autogluon.searcher.bayesopt.gpmxnet.constants import MCMCConfig, \
    OptimizationConfig
from autogluon.searcher.bayesopt.gpmxnet.gp_regression import \
    GaussianProcessRegression
from autogluon.searcher.bayesopt.gpmxnet.gpr_mcmc import GPRegressionMCMC
from autogluon.searcher.bayesopt.gpmxnet.kernel import Matern52, KernelFunction
from autogluon.searcher.bayesopt.gpmxnet.warping import WarpedKernel, Warping
from autogluon.searcher.bayesopt.gpmxnet.posterior_state import \
    GaussProcPosteriorState
from autogluon.searcher.bayesopt.datatypes.hp_ranges import \
    HyperparameterRanges, HyperparameterRangeCategorical, \
    HyperparameterRangeContinuous, HyperparameterRangeInteger
from autogluon.searcher.bayesopt.datatypes.tuning_job_state import \
    TuningJobState
from autogluon.searcher.bayesopt.datatypes.common import \
    FantasizedPendingEvaluation, Candidate, candidate_for_print
from autogluon.searcher.bayesopt.autogluon.hp_ranges import \
    HyperparameterRanges_CS
from autogluon.searcher.bayesopt.autogluon.gp_profiling import \
    GPMXNetSimpleProfiler
from autogluon.searcher.bayesopt.models.mxnet_base import SurrogateModelMXNet
from autogluon.searcher.bayesopt.autogluon.debug_log import DebugLogPrinter

logger = logging.getLogger(__name__)


GPModel = Union[GaussianProcessRegression, GPRegressionMCMC]


class InternalCandidateEvaluations(NamedTuple):
    X: np.ndarray
    y: np.ndarray
    mean: float
    std: float


# Note: If state.pending_evaluations is not empty, it must contain entries
# of type FantasizedPendingEvaluation, which contain the fantasy samples. This
# is the case only for internal states, the member GPMXNetModel.state has
# PendingEvaluation entries without the fantasy samples.
def get_internal_candidate_evaluations(
        state: TuningJobState, active_metric: str, normalize_targets: bool,
        num_fantasize_samples: int) -> InternalCandidateEvaluations:
    candidates_ndarray = []
    evaluation_values = []
    for candidate_evaluation in state.candidate_evaluations:
        candidates_ndarray.append(
            state.hp_ranges.to_ndarray(candidate_evaluation.candidate))
        evaluation_values.append(candidate_evaluation.metrics[active_metric])
    X = np.vstack(candidates_ndarray)
    # Normalize
    # Note: The fantasy values in state.pending_evaluations are sampled
    # from the model fit to normalized targets, so they are already
    # normalized
    y = np.vstack(evaluation_values).reshape((-1, 1))
    mean = 0.0
    std = 1.0
    if normalize_targets:
        std = max(np.std(y).item(), 1e-15)
        mean = np.mean(y).item()
        y = (y - mean) / std
    if state.pending_evaluations:
        # In this case, y becomes a matrix, where the observed values are
        # broadcasted
        fanta_lst = []
        cand_lst = []
        for pending_eval in state.pending_evaluations:
            assert isinstance(pending_eval, FantasizedPendingEvaluation), \
                "state.pending_evaluations has to contain FantasizedPendingEvaluation"
            fantasies = pending_eval.fantasies[active_metric]
            assert fantasies.size == num_fantasize_samples, \
                "All state.pending_evaluations entries must have length {}".format(
                    num_fantasize_samples)
            fanta_lst.append(fantasies.reshape((1, -1)))
            cand_lst.append(state.hp_ranges.to_ndarray(pending_eval.candidate))
        y = np.vstack([y * np.ones((1, num_fantasize_samples))] + fanta_lst)
        X = np.vstack([X] + cand_lst)
    return InternalCandidateEvaluations(X, y, mean, std)


def build_kernel(state: TuningJobState,
                 do_warping: bool = False) -> KernelFunction:
    dims, warping_ranges = dimensionality_and_warping_ranges(state.hp_ranges)
    kernel = Matern52(dims, ARD=True)
    if do_warping:
        return WarpedKernel(
            kernel=kernel, warping=Warping(dims, warping_ranges))
    else:
        return kernel


def default_gpmodel(
        state: TuningJobState, random_seed: int,
        optimization_config: OptimizationConfig) -> GaussianProcessRegression:
    return GaussianProcessRegression(
        kernel=build_kernel(state),
        optimization_config=optimization_config,
        random_seed=random_seed
    )


def default_gpmodel_mcmc(
        state: TuningJobState, random_seed: int,
        mcmc_config: MCMCConfig) -> GPRegressionMCMC:
    return GPRegressionMCMC(
        build_kernel=lambda: build_kernel(state),
        mcmc_config=mcmc_config,
        random_seed=random_seed
    )


def dimensionality_and_warping_ranges(hp_ranges: HyperparameterRanges) -> \
        Tuple[int, Dict[int, Tuple[float, float]]]:
    dims = 0
    warping_ranges = dict()
    # NOTE: This explicit loop over hp_ranges will fail if
    # HyperparameterRanges.hp_ranges is not implemented! Needs to be fixed if
    # it becomes an issue, either by moving the functionality here into
    # HyperparameterRanges, or by converting hp_ranges to
    # HyperparameterRanges_Impl, which supports the hp_ranges property.
    for hp_range in hp_ranges.hp_ranges:
        if not isinstance(hp_range, HyperparameterRangeCategorical):
            if isinstance(hp_range, HyperparameterRangeInteger):
                lower = int(round(hp_range.lower_bound))
                upper = int(round(hp_range.upper_bound))
            else:
                assert isinstance(hp_range, HyperparameterRangeContinuous)
                lower = float(hp_range.lower_bound)
                upper = float(hp_range.upper_bound)
            lower_internal = hp_range.to_ndarray(lower).item()
            upper_internal = hp_range.to_ndarray(upper).item()
            if upper_internal > lower_internal:  # exclude cases where max equal to min
                warping_ranges[dims] = (lower_internal, upper_internal)
            else:
                assert upper_internal == lower_internal
        dims += hp_range.ndarray_size()
    return dims, warping_ranges


class GPMXNetModel(SurrogateModelMXNet):
    def __init__(
            self, state: TuningJobState, active_metric: str, random_seed: int,
            gpmodel: GPModel, fit_parameters: bool, num_fantasy_samples: int,
            ctx_nd: mx.Context = None, dtype_nd = None,
            normalize_targets: bool = True,
            profiler: GPMXNetSimpleProfiler = None,
            debug_log: Optional[DebugLogPrinter] = None):
        """
        Given a TuningJobState state, the corresponding posterior state is
        computed here, based on which predictions are supported.
        Note: state is immutable. It must contain labeled examples.

        Parameters of the GP model in gpmodel are optimized iff fit_parameters
        is true. This requires state to contain labeled examples.

        We support pending evaluations via fantasizing. Note that state does
        not contain the fantasy values, but just the pending configs. Fantasy
        values are sampled here.

        :param state: TuningJobSubState
        :param active_metric: name of the metric to optimize.
        :param random_seed: Used only if GP model is created here
        :param gpmodel: GaussianProcessRegression model or GPRegressionMCMC model
        :param fit_parameters: Optimize parameters of gpmodel? Otherwise, these
            parameters are not changed
        :param num_fantasy_samples: See above
        :param normalize_targets: Normalize target values in
            state.candidate_evaluations?

        """
        super(GPMXNetModel, self).__init__(
            state, active_metric, random_seed, ctx_nd, dtype_nd, debug_log)
        assert num_fantasy_samples > 0
        self._gpmodel = gpmodel
        self.num_fantasy_samples = num_fantasy_samples
        self.normalize_targets = normalize_targets
        self.active_metric = active_metric
        # Compute posterior (including fitting (optional) and dealing with
        # pending evaluations)
        # If state.pending_evaluations is not empty, fantasy samples are drawn
        # here, but they are not maintained in the state (which is immutable),
        # and not in the posterior state either.
        # Instead, fantasy samples can be accessed via self.fantasy_samples.
        self.fantasy_samples = None
        self._compute_posterior(fit_parameters, profiler)

    def predict_nd(self, x_nd: mx.nd.NDArray) -> List[Tuple[mx.nd.NDArray, mx.nd.NDArray]]:
        """
        Note: Different to GPyOpt, means and stddevs are de-normalized here.
        """
        predictions_list_denormalized = []
        for posterior_mean, posterior_variance in self._gpmodel.predict(x_nd):
            assert posterior_mean.shape[0] == x_nd.shape[0], \
                (posterior_mean.shape, x_nd.shape)
            assert posterior_variance.shape == (x_nd.shape[0],), \
                (posterior_variance.shape, x_nd.shape)
            if self.state.pending_evaluations:
                # If there are pending candidates with fantasy values,
                # posterior_mean must be a matrix
                assert posterior_mean.ndim == 2 and \
                    posterior_mean.shape[1] == self.num_fantasy_samples, \
                    (posterior_mean.shape, self.num_fantasy_samples)
            predictions_list_denormalized.append(
                (posterior_mean * self.std + self.mean,
                mx.nd.sqrt(posterior_variance) * self.std))
        return predictions_list_denormalized

    @property
    def gpmodel(self) -> GPModel:
        return self._gpmodel

    def does_mcmc(self):
        return isinstance(self._gpmodel, GPRegressionMCMC)

    def posterior_states(self) -> Optional[List[GaussProcPosteriorState]]:
        return self._gpmodel.states

    def get_params(self):
        """
        Note: Once MCMC is supported, this method will have to be refactored.
        Note: If self.state still has no labeled data, the parameters returned
        are the initial ones, where an update would start from.

        :return: Hyperparameter dictionary
        """
        if not self.does_mcmc():
            return self._gpmodel.get_params()
        else:
            return dict()

    def set_params(self, param_dict):
        self._gpmodel.set_params(param_dict)

    def _compute_posterior(
            self, fit_parameters: bool, profiler: GPMXNetSimpleProfiler):
        """
        Completes __init__, by computing the posterior. If fit_parameters, this
        includes optimizing the surrogate model parameters.

        If self.state.pending_evaluations is not empty, we proceed as follows:
        - Compute posterior for state without pending evals
        - Draw fantasy values for pending evals
        - Recompute posterior (without fitting)

        """
        if self._debug_log is not None:
            self._debug_log.set_state(self.state)
        # Compute posterior for state without pending evals
        no_pending_state = self.state
        if self.state.pending_evaluations:
            no_pending_state = TuningJobState(
                hp_ranges=self.state.hp_ranges,
                candidate_evaluations=self.state.candidate_evaluations,
                failed_candidates=self.state.failed_candidates,
                pending_evaluations=[])
        self._posterior_for_state(no_pending_state, fit_parameters, profiler)
        if self.state.pending_evaluations:
            # Sample fantasy values for pending evals
            pending_configs = [
                x.candidate for x in self.state.pending_evaluations]
            new_pending = self._draw_fantasy_values(pending_configs)
            # Compute posterior for state with pending evals
            # Note: profiler is not passed here, this would overwrite the
            # results from the first call
            with_pending_state = TuningJobState(
                hp_ranges=self.state.hp_ranges,
                candidate_evaluations=self.state.candidate_evaluations,
                failed_candidates=self.state.failed_candidates,
                pending_evaluations=new_pending)
            self._posterior_for_state(
                with_pending_state, fit_parameters=False, profiler=None)
            # Note: At this point, the fantasy values are dropped, they are not
            # needed anymore. They've just been sampled for the posterior
            # computation. We still maintain them in self.fantasy_samples,
            # which is mainly used for testing
            self.fantasy_samples = new_pending

    def _posterior_for_state(
            self, state: TuningJobState, fit_parameters: bool,
            profiler: Optional[GPMXNetSimpleProfiler]):
        """
        Computes posterior for state.
        If fit_parameters and state.pending_evaluations is empty, we first
        optimize the model parameters.
        If state.pending_evaluations are given, these must be
        FantasizedPendingEvaluations, i.e. the fantasy values must have been
        sampled.

        """
        assert state.candidate_evaluations, \
            "Cannot compute posterior: state has no labeled datapoints"
        internal_candidate_evaluations = get_internal_candidate_evaluations(
            state, self.active_metric, self.normalize_targets,
            self.num_fantasy_samples)
        X_all = self.convert_np_to_nd(internal_candidate_evaluations.X)
        Y_all = self.convert_np_to_nd(internal_candidate_evaluations.y)
        assert X_all.shape[0] == Y_all.shape[0]
        self.mean = internal_candidate_evaluations.mean
        self.std = internal_candidate_evaluations.std

        fit_parameters = fit_parameters and (not state.pending_evaluations)
        if not fit_parameters:
            logger.info("Recomputing GP state")
            self._gpmodel.recompute_states(X_all, Y_all, profiler=profiler)
        else:
            logger.info("Fitting GP model")
            self._gpmodel.fit(X_all, Y_all, profiler=profiler)
        if self._debug_log is not None:
            self._debug_log.set_gp_params(self.get_params())
            if not state.pending_evaluations:
                deb_msg = "[GPMXNetModel._posterior_for_state]\n"
                deb_msg += ("- self.mean = {}\n".format(self.mean))
                deb_msg += ("- self.std = {}".format(self.std))
                logger.info(deb_msg)
                self._debug_log.set_targets(internal_candidate_evaluations.y)
            else:
                num_pending = len(state.pending_evaluations)
                fantasies = internal_candidate_evaluations.y[-num_pending:, :]
                self._debug_log.set_fantasies(fantasies)

    def _draw_fantasy_values(self, candidates: List[Candidate]) \
            -> List[FantasizedPendingEvaluation]:
        """
        Note: The fantasy values need not be de-normalized, because they are
        only used internally here (e.g., get_internal_candidate_evaluations).

        Note: A complication is that if the sampling methods of _gpmodel
        are called when there are no pending candidates (with fantasies) yet,
        they do return a single sample (instead of num_fantasy_samples). This
        is because GaussianProcessRegression knows about num_fantasy_samples
        only due to the form of the posterior state (bad design!).
        In this case, we draw num_fantasy_samples i.i.d.

        """
        if candidates:
            logger.debug("Fantasizing target values for candidates:\n{}"
                         .format(candidates))
            X_new = self.state.hp_ranges.to_ndarray_matrix(candidates)
            X_new_mx = self.convert_np_to_nd(X_new)
            # Special case (see header comment): If the current posterior state
            # does not contain pending candidates (no fantasies), we sample
            # num_fantasy_samples times i.i.d.
            num_samples = 1 if self._gpmodel.multiple_targets() \
                else self.num_fantasy_samples
            # We need joint sampling for >1 new candidates
            num_candidates = len(candidates)
            sample_func = self._gpmodel.sample_joint if num_candidates > 1 else \
                self._gpmodel.sample_marginals
            Y_new_mx = sample_func(X_new_mx, num_samples=num_samples)
            Y_new = Y_new_mx.asnumpy().astype(X_new.dtype, copy=False).reshape(
                (num_candidates, -1))
            return [
                FantasizedPendingEvaluation(
                    candidate, {self.active_metric: y_new.reshape((1, -1))})
                for candidate, y_new in zip(candidates, Y_new)
            ]
        else:
            return []

    def _current_best_filter_candidates(self, candidates):
        hp_ranges = self.state.hp_ranges
        if isinstance(hp_ranges, HyperparameterRanges_CS):
            candidates = hp_ranges.filter_for_last_pos_value(candidates)
            assert candidates, \
                "state.hp_ranges does not contain any candidates " + \
                "(labeled or pending) with resource attribute " + \
                "'{}' = {}".format(
                    hp_ranges.name_last_pos, hp_ranges.value_for_last_pos)
        return candidates
