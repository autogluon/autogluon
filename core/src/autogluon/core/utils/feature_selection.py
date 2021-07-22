from copy import deepcopy
import logging
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats
import time
from typing import Callable, Iterable, Sequence, Tuple
from .exceptions import TimeLimitExceeded

logger = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self, importance_fn: Callable[..., np.ndarray], importance_fn_args: dict, features: list, num_max_prune: int = 150) -> None:
        """
        Parameters
        ----------
        importance_fn : Callable
            Function that returns feature importance score of inputted features
        importance_fn_args : dict
            Parameters for importance_fn
        features : list
            List of feature names
        num_max_prune : int, default = 50
            Maximum number of features to consider pruning. Useful when there are too many features
            to assess individual feature importance scores even once.
        """
        self.importance_fn = importance_fn
        self.importance_fn_args = importance_fn_args
        if len(features) <= num_max_prune:
            self.prune_candidate_features = features
        else:
            self.prune_candidate_features = random.sample(features, k=num_max_prune)
        self.kept_features = [feature for feature in features if feature not in self.prune_candidate_features]

    def init_fi_dict(self) -> dict:
        raise NotImplementedError

    def compute_feature_importance(self, num_resource: int = 100, param_dict: dict = {}, threshold: int = 0.,
                                   time_limit: Tuple[None, float] = None, trajectory_plot_path: Tuple[None, str] = None) -> Tuple[dict, Sequence[dict]]:
        raise NotImplementedError

    def prune_features(self, param_dict: dict, threshold: float, prune_ratio: float = 1.) -> list:
        """
        Given param_dict and threshold, return a pruned list of features based on
        features in param_dict and threshold. Remove at most (prune_ratio * size of all features)
        features with lowest importance scores.
        """
        remaining_features = []
        pruned_features = []
        feature_scores = map(lambda info: {'feature': info[0], 'mu': info[1]['mu']}, param_dict.items())
        sorted_feature_scores = sorted(feature_scores, key=lambda feature_mean: feature_mean['mu'])
        max_prune_number = int(prune_ratio * (len(self.kept_features) + len(param_dict)))
        for info in sorted_feature_scores:
            feature, score = info['feature'], info['mu']
            if score > threshold or len(pruned_features) > max_prune_number:
                remaining_features.append(feature)
            else:
                pruned_features.append(feature)
        return self.kept_features + remaining_features

    def update_threshold(self, param_dict: dict, threshold: float, prune_ratio: float = 1.) -> float:
        """
        Return an updated threshold that would prune half as many features as what pruning with the
        inputted threshold would have accomplished. If inputted threshold feature prunes would have
        pruned one or less feature, return the original threshold. Only consider features in param_dict.
        """
        feature_means = map(lambda info: {'feature': info[0], 'mu': info[1]['mu']}, param_dict.items())
        sorted_feature_means = sorted(feature_means, key=lambda feature_mean: feature_mean['mu'])
        original_cutoff_index = 0
        for index, feature_mean in enumerate(sorted_feature_means):
            score = feature_mean['mu']
            if score <= threshold:
                original_cutoff_index = index
            else:
                break
        max_prune_number = int(prune_ratio * (len(self.kept_features) + len(param_dict)))
        # any feature whose index on sorted_feature_means is equal or lwer to this would have been dropped
        original_cutoff_index = min(original_cutoff_index, max_prune_number)
        if original_cutoff_index > 0:
            new_cutoff_index = original_cutoff_index//2
            threshold = sorted_feature_means[new_cutoff_index]['mu']
        return threshold


class UniformFeatureSelector(FeatureSelector):
    def __init__(self, importance_fn: Callable[..., np.ndarray], importance_fn_args: dict, features: list, num_max_prune: int = 100) -> None:
        super().__init__(importance_fn, importance_fn_args, features, num_max_prune=num_max_prune)

    def compute_feature_importance(self, num_resource: int = 100, param_dict: dict = {}, threshold: int = 0.,
                                   time_limit: Tuple[None, float] = None, trajectory_plot_path: Tuple[None, str] = None) -> Tuple[dict, Sequence[dict]]:
        """
        Uniformly allocate feature importance computation across features in self.prune_candidate_features.
        Return resulting param_dict with mean feature importance score estimate per feature in self.prune_candidate_features.
        If param_dict already exists, the existing stats are combined with new ones.
        """
        shuffles_per_feature = num_resource // len(self.prune_candidate_features)
        shuffles_left = num_resource % len(self.prune_candidate_features)
        importance_fn_args = deepcopy(self.importance_fn_args)
        importance_fn_args['features'] = self.prune_candidate_features
        importance_fn_args['num_shuffle_sets'] = shuffles_per_feature
        importance_df = self.importance_fn(**importance_fn_args)
        if shuffles_left > 0:
            importance_fn_args['features'] = self.prune_candidate_features[:shuffles_left]
            importance_fn_args['num_shuffle_sets'] = 1
            subset_importance_df = self.importance_fn(**importance_fn_args)
        else:
            subset_importance_df = None

        for feature, info in importance_df.iterrows():
            if subset_importance_df is not None and feature in subset_importance_df.index:
                num_resource = shuffles_per_feature + 1
                mean_importance = (info['importance']*info['n']+subset_importance_df['importance'][feature])/num_resource
            else:
                num_resource = shuffles_per_feature
                mean_importance = info['importance']
            if feature in param_dict:
                # If param_dict already exists, update importance estimates with current run values
                prev_importance = param_dict[feature]['importance']
                prev_num_resource = param_dict[feature]['num_resource']
                mean_importance = (mean_importance*num_resource + prev_importance*prev_num_resource)/(num_resource + prev_num_resource)
                num_resource = num_resource + prev_num_resource
            param_dict[feature] = {'mu': mean_importance, 'num_resource': num_resource}
        return param_dict, [param_dict]


class BayesianFeatureSelector(FeatureSelector):
    def compute_feature_importance(self, num_resource: int = 100, param_dict: dict = {}, threshold: int = 0.,
                                   time_limit: Tuple[None, float] = None, trajectory_plot_path: Tuple[None, str] = None) -> Tuple[dict, Sequence[dict]]:
        """
        Efficiently compute feature importance based on feature importance threshold by making use
        of `num_resource` feature importance computations. Return resulting param_dict with mean feature
        importance score estimate per feature in self.prune_candidate_features.

        # TODO: Change this to be more flexible with resource
        """
        if param_dict == {}:
            param_dict = self.init_fi_dict()
        # Every iteration, save prior mean and sigma and use them to plot sequence of normal pdfs for each feature
        trajectories = {feature: [] for feature in param_dict.keys()}
        for feature in param_dict.keys():
            trajectories[feature].append(param_dict[feature])

        for iteration in range(num_resource):
            expected_utilities = {}
            for feature in self.prune_candidate_features:
                utility = self.compute_expected_utility(param_dict[feature], threshold)
                expected_utilities[feature] = utility
            sorted_expected_utilities = sorted(expected_utilities.items(), key=lambda el: el[1])
            promising_feature = sorted_expected_utilities[-1][0]
            time_start = time.time()
            importance_fn_args = deepcopy(self.importance_fn_args)
            importance_fn_args['features'] = [promising_feature]
            importance_df = self.importance_fn(**importance_fn_args)
            updated_params = self.bayes_update(importance_df['importance'], param_dict[promising_feature])
            param_dict[promising_feature] = {param: value[0] if isinstance(value, Iterable) else value for param, value in updated_params.items()}
            param_dict[promising_feature]['num_resource'] += 1
            param_dict[promising_feature]['latest_pull_iter'] = iteration
            if time_limit and time.time() - time_start >= time_limit:
                raise TimeLimitExceeded
            for feature in param_dict.keys():
                trajectories[feature].append(param_dict[feature])

        if trajectory_plot_path:
            self.plot_trajectories(trajectories, trajectory_plot_path)
        return param_dict, trajectories

    def compute_expected_utility(self, param_dict: dict, threshold: float, n_sample: int = 1000):
        raise NotImplementedError

    def bayes_update(self, obs: np.ndarray, param_dict: dict):
        raise NotImplementedError


class NormalFeatureSelector(BayesianFeatureSelector):
    def __init__(self, importance_fn: Callable[..., np.ndarray], importance_fn_args: dict, features: list, num_max_prune: int = 100) -> None:
        super().__init__(importance_fn, importance_fn_args, features, num_max_prune=num_max_prune)
        self.prior_mu = 0.
        self.prior_sigma = 0.05
        self.obs_sigma = 0.01

    def init_fi_dict(self) -> dict:
        """
        Initialize param_dict with self.prune_candidates
        """
        param_dict = {}
        for feature in self.prune_candidate_features:
            param_dict[feature] = {
                'mu': self.prior_mu,
                'sigma': self.prior_sigma,
                'obs_sigma': self.obs_sigma,
                'num_resource': 0,
                'latest_pull_iter': -1,
            }
        return param_dict

    def compute_expected_utility(self, param_dict: dict, threshold: float, n_sample: int = 1000) -> float:
        """
        Return expected information gain on whether feature is relevant for a single feature
        under the current model.
        - Compute p(X=1) using prior normal CDF and with it calculate prior binary entropy
        - Sample 1000 samples from normal distribution
        - For each sample, compute p(X=0|s) and with it calculate posterior binary entropy
        - Average posterior binary entropy from all samples and subtract from prior binary entropy
        """
        mu, sigma, obs_sigma = param_dict['mu'], param_dict['sigma'], param_dict['obs_sigma']
        prior_prob = stats.norm.cdf(threshold, mu, sigma)
        prior_entropy = stats.bernoulli.entropy(prior_prob)
        marginal_samples = stats.norm.rvs(size=n_sample, loc=mu, scale=sigma)
        posteriors = self.bayes_update(marginal_samples, param_dict)
        posterior_probs = stats.norm.cdf(threshold, posteriors['mu'], posteriors['sigma'])
        posterior_entropys = stats.bernoulli.entropy(posterior_probs)
        expected_posterior_entropy = posterior_entropys.mean()
        return prior_entropy - expected_posterior_entropy

    def bayes_update(self, obs: np.ndarray, param_dict: dict) -> dict:
        """
        Batched Normal-Normal conjugate Bayes update for a single observation.

        Parameters
        ----------
        obs : np.ndarray
            1D list of IID observations where each observation generates a different posterior distribution.
        param_dict : dict
            Dictionary containing prior mean, prior standard deviation, and observation standard deviation.
        """
        mu, sigma, obs_sigma = param_dict['mu'], param_dict['sigma'], param_dict['obs_sigma']
        posterior_sigmas = np.sqrt(1 / (1/sigma**2 + 1/obs_sigma**2))
        posterior_mus = posterior_sigmas**2 * (mu/sigma**2 + obs/obs_sigma**2)
        # posterior_sigma = np.sqrt(1 / (1/sigma**2 + len(obs)/obs_sigma**2))
        # posterior_mu = posterior_sigma**2 * (mu/sigma**2 + sum(obs)/obs_sigma**2)
        result = deepcopy(param_dict)
        result['mu'] = posterior_mus
        result['sigma'] = posterior_sigmas
        return result

    def plot_trajectories(self, trajectories: dict, x_len=5, x_lo=-0.1, x_hi=0.1):
        y_len = math.ceil(len(trajectories) / x_len)
        fig, axs = plt.subplots(y_len, x_len, figsize=(x_len*3, y_len*3), squeeze=False)
        x = np.linspace(x_lo, x_hi, 100)
        # sort trajectories in ascending order based on final mean score
        trajectories = sorted(trajectories.items(), key=lambda el: el[1][-1]['mu'])
        for i, info in enumerate(trajectories):
            feature, trajectory = info
            for param in trajectory:
                axs[i // x_len, i % x_len].plot(x, stats.norm.pdf(x, param['mu'], param['sigma']))
            n_unique = len(set(map(lambda param: tuple(param.values()), trajectory)))
            axs[i // x_len, i % x_len].set_title(f"{feature} ({round(trajectory[-1]['mu'],4)},{n_unique})")
        fig.suptitle(f"Prior Evolution Over {len(trajectory)-1} Timesteps (FI Score, # Updates)")
        fig.tight_layout()
        fig.savefig(f"normal_{int(time.time())}.png")
