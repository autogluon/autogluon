import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..tuning_algorithms.base_classes import SurrogateModel, AcquisitionFunction, \
    OutputSurrogateModel, assign_active_metric, dictionarize_objective


@dataclass
class HeadWithGradient:
    hvals: np.array
    dh_dmean: Dict[str, np.array]  # Maps each output model to its head gradient wrt to the predictive mean
    dh_dstd: Dict[str, np.array]  # Maps each output model to its head gradient wrt to the predictive std


class MeanStdAcquisitionFunction(AcquisitionFunction, ABC):
    """
    Base class for standard acquisition functions which depend on predictive
    mean and stddev. Subclasses have to implement the head and its derivatives
    w.r.t. mean and std:

        f(x, model) = h(mean, std, model.current_best())

    If model is a SurrogateModel, then active_metric is ignored. If model is a Dict mapping output names to models,
    then active_metric must be given.

    NOTE that acquisition functions will always be *minimized*!

    """
    def __init__(self, model: OutputSurrogateModel, active_metric: str = None):
        super().__init__(model, active_metric)
        if isinstance(model, SurrogateModel):
            # Ignore active_metric
            model = dictionarize_objective(model)
        assert isinstance(model, Dict)
        self.model = model
        self.model_output_names = sorted(self.model.keys())
        for output_model in self.model.values():
            assert 'mean' in output_model.keys_predict()
            assert 'std' in output_model.keys_predict()
        self.active_metric = assign_active_metric(self.model, active_metric)

    def compute_acq(self, inputs: np.ndarray, model: Optional[OutputSurrogateModel] = None) -> np.ndarray:
        if model is None:
            model = self.model
        if isinstance(model, SurrogateModel):
            self.model = dictionarize_objective(model)
        if inputs.ndim == 1:
            inputs = inputs.reshape((1, -1))
        output_to_predictions = self._map_outputs_to_predictions(inputs)
        self.num_mcmc_samples = len(output_to_predictions[self.active_metric])  # a single sample means no MCMC
        current_best_list = self._get_current_best_for_active_metric()

        fvals_list = []  # this will contain the acquisition function for each MCMC sample (if any)
        # Loop over MCMC samples
        for predictions, current_best in zip(
                zip(*output_to_predictions.values()),
                current_best_list):
            # Dictionaries mapping each output model name to predictions and current_bests for the current MCMC sample
            output_model_to_mean_std = dict(zip(self.model_output_names, predictions))
            # Create a dictionary mapping each output model name to means, stds and current best,
            # all three in a shape that can be fed into self._compute_heads
            output_to_mean_std = {}
            for output_model, prediction in output_model_to_mean_std.items():
                means = prediction['mean']
                stds = prediction['std']
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
                output_to_mean_std[output_model] = {'mean': means, 'std': stds}
            # Compute the acquisition function value
            fvals = self._compute_heads(output_to_mean_std, current_best)
            # Average over fantasies if there are any
            if num_fantasies > 1:
                fvals = np.mean(fvals, axis=1)
            fvals_list.append(fvals)

        return np.mean(fvals_list, axis=0)

    def compute_acq_with_gradient(self, input: np.ndarray, model: Optional[OutputSurrogateModel] = None) -> \
            Tuple[float, np.ndarray]:
        if model is None:
            model = self.model
        if isinstance(model, SurrogateModel):
            self.model = dictionarize_objective(model)
        output_to_predictions = self._map_outputs_to_predictions(input.reshape(1, -1))
        self.num_mcmc_samples = len(output_to_predictions[self.active_metric])  # a single sample means no MCMC
        current_best_list = self._get_current_best_for_active_metric()

        fvals = []  # this will contain the value of the acquisition function for each MCMC sample (if any)
        output_to_head_gradients = {output_name: [] for output_name in self.model_output_names}  # this dictionary will
        # map each output model to its dictionary of head gradients (one dictionary of head gradients per MCMC sample)
        # Loop over MCMC samples (if any)
        for predictions, current_best in zip(
                zip(*output_to_predictions.values()),
                current_best_list):
            output_model_to_mean_std = dict(zip(self.model_output_names, predictions))
            # Create a dictionary mapping each output model name to means, stds and current best,
            # all three in a shape that can be fed into self._compute_heads_and_gradient
            output_to_mean_std = {}
            for output_name, prediction in output_model_to_mean_std.items():
                mean = prediction['mean'].reshape((-1,))
                num_fantasies = mean.size
                std = prediction['std'].reshape((1,))
                if current_best is not None:
                    assert current_best.size == num_fantasies
                    current_best = current_best.reshape((-1,))
                output_to_mean_std[output_name] = {'mean': mean, 'std': std}
            head_result = self._compute_head_and_gradient(output_to_mean_std, current_best)
            fvals.append(np.mean(head_result.hvals))

            for output_name in self.model_output_names:
                # Fill up output_to_head_gradients, which maps each output_name to its head gradients. Head gradients
                # are a list: each list element is an MCMC sample and contains the head gradient of the mean and std.

                # Each head_gradient is composed for backward_gradient call. This involves reshaping. Also, we are
                # averaging over fantasies (if num_fantasies > 1), which is not done in _compute_head_and_gradient
                # (dh_dmean, dh_dstd have the same shape as mean), so we have to divide by num_fantasies.
                output_mean = output_model_to_mean_std[output_name]['mean']
                output_std = output_model_to_mean_std[output_name]['std']
                num_fantasies = output_mean.size
                head_gradient = {'mean': head_result.dh_dmean[output_name].reshape(
                    output_mean.shape) / num_fantasies,
                                 'std': np.array([np.mean(head_result.dh_dstd[output_name])]).reshape(
                                     output_std.shape)}
                output_to_head_gradients[output_name].append(head_gradient)

        fval = np.mean(fvals)  # Averaging over MCMC samples (if any)
        gradient = 0.0
        # Sum up the gradients coming from each output model
        for output_name, output_model in model.items():
            # Gradients are computed by the model
            gradient_list = output_model.backward_gradient(input, output_to_head_gradients[output_name])
            output_gradient = np.mean(gradient_list, axis=0)  # Averaging over MCMC samples (if any)
            gradient += output_gradient
        return fval, gradient

    def _map_outputs_to_predictions(self, inputs):
        """
        Returns a dictionary mapping each output to the predictive mean and std at the given inputs.
        """
        output_to_predictions = {}
        for output_name, output_model in self.model.items():
            assert 'mean' in output_model.keys_predict()
            assert 'std' in output_model.keys_predict()
            output_to_predictions[output_name] = output_model.predict(inputs)
        return output_to_predictions

    def _get_current_best_for_active_metric(self):
        """
        Returns a list of current best, one per MCMC sample (if head requires the current best).
        """
        if self._head_needs_current_best():
            current_best_list = self.model[self.active_metric].current_best()  # list of  1 x nf arrays
        else:
            current_best_list = [None] * self.num_mcmc_samples
        return current_best_list

    def _extract_active_metric_stats(self, output_to_mean_std):
        mean = output_to_mean_std[self.active_metric]['mean']
        stds = output_to_mean_std[self.active_metric]['std']
        return mean, stds

    @abstractmethod
    def _head_needs_current_best(self) -> bool:
        """
        :return: Is the current_best argument in _compute_head needed?
        """
        pass

    @abstractmethod
    def _compute_heads(
            self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
            current_best: Optional[np.ndarray]) -> np.ndarray:
        """
        If mean has >1 columns, both std and current_best are supposed to be
        broadcasted. The return value has the same shape as mean.

        :param: output_to_mean_std_best: Dictionary mapping each output to a dictionary containing:
                                        mean: Predictive means, shape (n, nf) or (n,)
                                        std: Predictive stddevs, shape (n, 1) or (n,)
                current_best: Incumbent, shape (1, nf) or (1,)
        :return: h(means, stds, current_best), same shape as means
        """
        pass

    @abstractmethod
    def _compute_head_and_gradient(
            self, output_to_mean_std: Dict[str, Dict[str, np.ndarray]],
            current_best: Optional[np.ndarray]) -> HeadWithGradient:
        """
        Computes both head value and head gradients, for a single input.

        :param: output_to_mean_std: Dictionary mapping each output to a dictionary containing:
                                    mean: Predictive mean, shape (nf,)
                                    std: Predictive stddev, shape (1,)
                current_best: Incumbent, shape (nf,)
        :return: HeadWithGradient containing hvals and, for each output model, dh_dmean, dh_dstd.
                 All HeadWithGradient values have the same shape as mean

        """
        pass
