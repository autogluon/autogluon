import numpy as np
from typing import Dict, Optional

import logging
from .meanstd_acqfunc import MeanStdAcquisitionFunction, AcquisitionWithMultiModelCurrentBest, HeadWithGradient
from scipy.stats import norm
from ..tuning_algorithms.base_classes import OutputSurrogateModel, SurrogateModel
from ..utils.density import get_quantiles


logger = logging.getLogger(__name__)
MIN_COST = 1e-12   # For numerical stability when dividing EI / cost
MIN_STD_CONSTRAINT = 1e-12   # For numerical stability when computing the constraint probability in CEI


def _extract_active_and_secondary_metric(model_output_names, active_metric):
    """
    Returns the active metric and the secondary metric (such as the cost or constraint metric) from model_output_names.
    """

    assert len(model_output_names) == 2, f"The model should consist of exactly 2 outputs, " \
                                         f"while the current outputs are {model_output_names}"
    assert active_metric in model_output_names, f"{active_metric} is not a valid metric. " \
                                                f"The metric name must match one of the following metrics " \
                                                f"in the model output: {model_output_names}"
    if model_output_names[0] == active_metric:
        secondary_metric = model_output_names[1]
    else:
        secondary_metric = model_output_names[0]
    logger.info(f"There are two metrics in the output: {model_output_names}. "
                f"The metric to optimize was set to '{active_metric}'. "
                f"The secondary metric is assumed to be '{secondary_metric}'")
    return active_metric, secondary_metric


class EIAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Minus expected improvement acquisition function
    (minus because the convention is to always minimize acquisition functions)

    """
    def __init__(self, model: OutputSurrogateModel, active_metric: str = None,
                 jitter: float = 0.01):
        assert isinstance(model, SurrogateModel)
        super().__init__(model, active_metric)
        self.jitter = jitter

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_heads(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                       current_best: Optional[np.ndarray]) -> np.ndarray:
        means, stds = self._extract_active_metric_stats(output_to_mean_std)
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        return (-stds) * (u * Phi + phi)

    def _compute_head_and_gradient(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                                   current_best: Optional[np.ndarray]) -> HeadWithGradient:
        mean, std = self._extract_active_metric_stats(output_to_mean_std)
        assert current_best is not None

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        f_acqu = std * (u * Phi + phi)
        return HeadWithGradient(
            hvals=-f_acqu,
            dh_dmean={self.active_metric: Phi},
            dh_dstd={self.active_metric: -phi})


class LCBAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Lower confidence bound (LCB) acquisition function:

        h(mean, std) = mean - kappa * std

    """
    def __init__(self, model: OutputSurrogateModel, kappa: float, active_metric: str = None):
        super().__init__(model, active_metric)
        assert kappa > 0, 'kappa must be positive'
        self.kappa = kappa

    def _head_needs_current_best(self) -> bool:
        return False

    def _compute_heads(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                       current_best: Optional[np.ndarray]) -> np.ndarray:
        means, stds, _ = self._extract_active_metric_stats(output_to_mean_std)
        return means - stds * self.kappa

    def _compute_head_and_gradient(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                                   current_best: Optional[np.ndarray]) -> HeadWithGradient:
        mean, std = self._extract_active_metric_stats(output_to_mean_std)
        ones_like_mean = np.ones_like(mean)
        ones_like_std = np.ones_like(std)
        return HeadWithGradient(
            hvals=mean - std * self.kappa,
            dh_dmean={self.active_metric: ones_like_mean},
            dh_dstd={self.active_metric: (-self.kappa) * ones_like_std})


class EIpuAcquisitionFunction(MeanStdAcquisitionFunction):
    """
    Minus cost-aware expected improvement acquisition function.
    (minus because the convention is to always minimize the acquisition function)

    This is defined as EIpu(x) = EI(x) / cost(x), where cost(x) is the predictive mean of the cost model at x.

    Note: two metrics are expected in the model output: the main objective and the cost.
    The main objective needs to be indicated as active_metric when initializing EIpuAcquisitionFunction.
    The cost is automatically assumed to be the other metric.

    """
    def __init__(self, model: OutputSurrogateModel, active_metric: str, jitter: float = 0.01):
        super(EIpuAcquisitionFunction, self).__init__(model, active_metric)
        self.jitter = jitter
        self.active_metric, self.cost_metric = _extract_active_and_secondary_metric(
            self.model_output_names, active_metric)

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_heads(
            self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
            current_best: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns minus the cost-aware expected improvement.
        """
        means, stds = self._extract_active_metric_stats(output_to_mean_std)
        assert current_best is not None
        pred_costs = self._extract_positive_cost(output_to_mean_std)

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        f_ei = stds * (u * Phi + phi)
        f_acqu = f_ei / pred_costs
        return - f_acqu

    def _compute_head_and_gradient(
            self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
            current_best: Optional[np.ndarray]) -> HeadWithGradient:
        """
        Returns minus cost-aware expected improvement and, for each output model, the gradients
        with respect to the mean and standard deviation of that model.
        """
        mean, std = self._extract_active_metric_stats(output_to_mean_std)
        assert current_best is not None
        pred_cost = self._extract_positive_cost(output_to_mean_std)

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)
        f_ei = std * (u * Phi + phi)
        f_acqu = f_ei / pred_cost

        dh_dmean_active = Phi / pred_cost
        dh_dstd_active = - phi / pred_cost
        dh_dmean_cost = f_ei / (pred_cost ** 2)  # We flip the sign twice: once because of the derivative of 1 / x
        # and once because the head is actually - f_ei
        dh_dstd_cost = np.zeros_like(dh_dstd_active)   # EIpu does not depend on the standard deviation of cost

        return HeadWithGradient(
            hvals=-f_acqu,
            dh_dmean={self.active_metric: dh_dmean_active, self.cost_metric: dh_dmean_cost},
            dh_dstd={self.active_metric: dh_dstd_active, self.cost_metric: dh_dstd_cost}
        )

    def _extract_positive_cost(self, output_to_mean_std_best):
        pred_cost = output_to_mean_std_best[self.cost_metric]['mean']
        if any(pred_cost) < 0.0:
            logger.warning(f'The model for {self.cost_metric} predicted some negative cost. '
                           f'Capping the minimum cost at {MIN_COST}.')
        pred_cost = np.maximum(pred_cost, MIN_COST)  # ensure that the predicted cost/run-time is positive
        return pred_cost


class CEIAcquisitionFunction(AcquisitionWithMultiModelCurrentBest):
    """
    Minus constrained expected improvement acquisition function.
    (minus because the convention is to always minimize the acquisition function)

    This is defined as CEI(x) = EI(x) * P(c(x) <= 0), where EI is the standard expected improvement with respect
    to the current *feasible best*, and P(c(x) <= 0) is the probability that the hyperparameter
    configuration x satisfies the constraint modeled by c(x).

    If there are no feasible hyperparameters yet, the current feasible best is undefined. Thus, CEI is
    reduced to the P(c(x) <= 0) term until a feasible configuration is found.

    Two metrics are expected in the model output: the main objective and the constraint metric.
    The main objective needs to be indicated as active_metric when initializing CEIAcquisitionFunction.
    The constraint is automatically assumed to be the other metric.

    References on CEI:
    Gardner et al., Bayesian Optimization with Inequality Constraints. In ICML, 2014.
    Gelbart et al., Bayesian Optimization with Unknown Constraints. In UAI, 2014.

    """
    def __init__(self, model: OutputSurrogateModel, active_metric: str, jitter: float = 0.01):
        super(CEIAcquisitionFunction, self).__init__(model, active_metric)
        self.jitter = jitter
        self._feasible_best_list = None
        self.active_metric, self.constraint_metric = _extract_active_and_secondary_metric(
            self.model_output_names, active_metric)

    def _head_needs_current_best(self) -> bool:
        return True

    def _compute_heads(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                       current_best: Optional[np.ndarray]) -> np.ndarray:
        """
        Returns minus the constrained expected improvement (- CEI).
        """
        assert current_best is not None
        means, stds = self._extract_active_metric_stats(output_to_mean_std)
        means_constr = output_to_mean_std[self.constraint_metric]['mean']
        stds_constr = output_to_mean_std[self.constraint_metric]['std']
        # Compute the probability of satisfying the constraint P(c(x) <= 0)
        constr_probs = norm.cdf(- means_constr / (stds_constr + MIN_STD_CONSTRAINT))
        # If for some fantasies there are not feasible candidates, there is also no current_best (i.e., a nan).
        # The acquisition function is replaced by only the P(c(x) <= 0) term when no feasible best exist.
        feas_idx = ~np.isnan(current_best).flatten()
        num_fantasies = current_best.size
        means = means.reshape((-1, num_fantasies))
        stds = stds.reshape((-1, 1))
        current_best = current_best.reshape((1, num_fantasies))
        constr_probs = constr_probs.reshape((-1, num_fantasies))

        # phi, Phi is PDF and CDF of Gaussian
        phi, Phi, u = get_quantiles(self.jitter, current_best, means, stds)
        f_ei = stds * (u * Phi + phi)
        # CEI(x) = EI(x) * P(c(x) <= 0) if feasible best exists, CEI(x) = P(c(x) <= 0) otherwise
        f_acqu = np.where(feas_idx, f_ei * constr_probs, constr_probs)
        return - f_acqu

    def _compute_head_and_gradient(self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
                                   current_best: Optional[np.ndarray]) -> HeadWithGradient:
        """
        Returns minus cost-aware expected improvement (- CEI) and, for each output model, the gradients
        with respect to the mean and standard deviation of that model.
        """
        assert current_best is not None
        mean, std = self._extract_active_metric_stats(output_to_mean_std)
        mean_constr = output_to_mean_std[self.constraint_metric]['mean']
        std_constr = output_to_mean_std[self.constraint_metric]['std']
        # Compute the probability of satisfying the constraint P(c(x) <= 0)
        std_constr = std_constr + MIN_STD_CONSTRAINT
        z = - mean_constr / std_constr
        constr_prob = norm.cdf(z)
        # Useful variables for computing the head gradients
        mean_over_squared_std_constr = mean_constr / std_constr ** 2
        inverse_std_constr = 1. / std_constr
        phi_constr = norm.pdf(z)

        # If for some fantasies there are not feasible candidates, there is also no current_best (i.e., a nan).
        # The acquisition function is replaced by only the P(c(x) <= 0) term when no feasible best exist.
        feas_idx = ~np.isnan(current_best)
        phi, Phi, u = get_quantiles(self.jitter, current_best, mean, std)  # phi, Phi is PDF and CDF of Gaussian
        f_ei = std * (u * Phi + phi)
        f_acqu = np.where(feas_idx, f_ei * constr_prob, constr_prob)  # CEI(x) = EI(x) * P(c(x) <= 0) if feasible best
        # exists, CEI(x) = P(c(x) <= 0) otherwise
        dh_dmean_constraint_feas = f_ei * inverse_std_constr * phi_constr
        dh_dstd_constraint_feas = - f_ei * mean_over_squared_std_constr * phi_constr
        dh_dmean_active_feas = Phi * constr_prob
        dh_dstd_active_feas = - phi * constr_prob
        dh_dmean_constraint_infeas = inverse_std_constr * phi_constr
        dh_dstd_constraint_infeas = - mean_over_squared_std_constr * phi_constr
        dh_dmean_active_infeas = np.zeros_like(phi_constr)
        dh_dstd_active_infeas = np.zeros_like(phi_constr)
        dh_dmean_active = np.where(feas_idx, dh_dmean_active_feas, dh_dmean_active_infeas)
        dh_dstd_active = np.where(feas_idx, dh_dstd_active_feas, dh_dstd_active_infeas)
        dh_dmean_constraint = np.where(feas_idx, dh_dmean_constraint_feas, dh_dmean_constraint_infeas)
        dh_dstd_constraint = np.where(feas_idx, dh_dstd_constraint_feas, dh_dstd_constraint_infeas)
        return HeadWithGradient(
            hvals=-f_acqu,
            dh_dmean={self.active_metric: dh_dmean_active, self.constraint_metric: dh_dmean_constraint},
            dh_dstd={self.active_metric: dh_dstd_active, self.constraint_metric: dh_dstd_constraint}
        )

    def _get_current_best_for_active_metric(self):
        """
        Returns a list of current best, one per MCMC sample (if head requires the current best).

        Filters out the infeasible candidates when computing the current best. This is needed as the constrained
        expected improvement uses the current best over the *feasible* hyperparameter
        configurations (i.e., satisfying the constraint) and not over all hyperparameter configurations.

        The assumption is that a hyperparameter configuration is feasible iff the predictive mean of the
        constraint model is <= 0 at that configuration.
        """
        if self._head_needs_current_best():
            if self._feasible_best_list is not None:
                feasible_best_list = self._feasible_best_list
            else:
                models_to_means = self._map_models_to_candidate_predictive_means()
                assert len(models_to_means) == 2, 'The model should consist of exactly 2 outputs.'
                all_means_active = models_to_means[self.active_metric]
                all_means_constraint = models_to_means[self.constraint_metric]
                assert len(all_means_constraint) == len(all_means_constraint), \
                    'All models must have the same number of MCMC samples.'
                feasible_best_list = []
                # Loop over MCMC samples (if any)
                for means_active, means_constraint in zip(all_means_active, all_means_constraint):
                    assert means_active.shape == means_constraint.shape, \
                        'The predictive means from each model must have the same shape.'
                    # Remove all infeasible candidates (i.e., where means_constraint is >= 0)
                    means_active[means_constraint >= 0] = np.nan
                    # Compute the current *feasible* best (separately for every fantasy)
                    min_across_observations = np.nanmin(means_active, axis=0)
                    feasible_best_list.append(min_across_observations)
                    self._feasible_best_list = feasible_best_list
        else:
            feasible_best_list = [None] * self.num_mcmc_samples
        return feasible_best_list
