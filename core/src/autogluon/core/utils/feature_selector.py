from copy import deepcopy
import numpy as np
import pandas as pd
import time
from typing import Callable, Sequence, Tuple, Union
from .exceptions import TimeLimitExceeded


class FeatureImportanceHelper:
    def __init__(self, importance_fn: Callable[..., np.ndarray], importance_fn_args: dict, features: Sequence[str], golden_features: Sequence[str] = []):
        """
        Parameters
        ----------
        importance_fn : Callable
            Function that returns feature importance score of inputted features
        importance_fn_args : dict
            Parameters for importance_fn
        features : Sequence[str]
            List of feature names
        kept_features : Sequence[str]
            List of feature names to never compute feature importance for (example use: golden features)
        """
        self.importance_fn = importance_fn
        self.importance_fn_args = importance_fn_args
        self.golden_features = golden_features
        self.prune_candidate_features = [feature for feature in features if feature not in golden_features]
        self.custom_header = "AUTOGLUONPRUNE"
        self.custom_separator = "//"

    def compute_fi(self, num_resource: int, **kwargs) -> Tuple[pd.DataFrame, Sequence[pd.DataFrame]]:
        raise NotImplementedError


class UniformFeatureImportanceHelper(FeatureImportanceHelper):
    def compute_fi(self, num_resource: int, **kwargs) -> Tuple[pd.DataFrame, Sequence[pd.DataFrame]]:
        """
        Uniformly allocate feature importance computation across features in self.prune_candidate_features.
        Return resulting param_dict with mean feature importance score estimate per feature in self.prune_candidate_features.
        If param_dict already exists, the existing stats are combined with new ones.
        """
        time_limit = kwargs.get('time_limit', None)
        prev_importance_df = kwargs.get('prev_importance_df', None)
        shuffles_per_feature = num_resource // len(self.prune_candidate_features)
        shuffles_left = num_resource % len(self.prune_candidate_features)
        importance_fn_args = deepcopy(self.importance_fn_args)
        importance_fn_args['features'] = self.prune_candidate_features
        importance_fn_args['num_shuffle_sets'] = shuffles_per_feature
        importance_fn_args['time_limit'] = time_limit
        time_start = time.time()
        importance_df = self.importance_fn(**importance_fn_args)
        if shuffles_left > 0:
            importance_fn_args = deepcopy(self.importance_fn_args)
            importance_fn_args['features'] = self.prune_candidate_features[:shuffles_left]
            importance_fn_args['num_shuffle_sets'] = 1
            importance_fn_args['time_limit'] = time_limit
            subset_importance_df = self.importance_fn(**importance_fn_args)
        else:
            subset_importance_df = None
        time_elapsed = time.time() - time_start
        if time_limit and time_limit - time_elapsed <= 0:
            raise TimeLimitExceeded

        result = {'feature': [], 'importance': [], 'n': []}
        for feature, info in importance_df.iterrows():
            if subset_importance_df is not None and feature in subset_importance_df.index:
                num_resource = shuffles_per_feature + 1
                mean_importance = (info['importance']*info['n']+subset_importance_df['importance'][feature])/num_resource
            else:
                num_resource = shuffles_per_feature
                mean_importance = info['importance']
            if prev_importance_df is not None and feature in prev_importance_df.index:
                # If param_dict already exists, update importance estimates with current run values
                prev_importance = prev_importance_df[feature]['importance']
                prev_num_resource = prev_importance_df[feature]['num_resource']
                mean_importance = (mean_importance*num_resource + prev_importance*prev_num_resource)/(num_resource + prev_num_resource)
                num_resource = num_resource + prev_num_resource
            result['feature'].append(feature)
            result['importance'].append(mean_importance)
            result['n'].append(num_resource)
        result_df = pd.DataFrame(result).set_index(['feature'])
        return result_df, [result_df]


class BackwardSearchFeatureImportanceHelper(FeatureImportanceHelper):
    def __init__(self, prune_ratio: float, **kwargs):
        super().__init__(**kwargs)
        self.prune_ratio = prune_ratio

    def compute_fi(self, num_resource: int, **kwargs) -> Tuple[pd.DataFrame, Sequence[pd.DataFrame]]:
        """
        Perform backward search based on feature importance score. Every iteration, assess |features| candidate feature subsets
        generated by appending a leftover feature to previous lowest feature importance feature subset. Each feature subset
        receives max(1, num_resource / (|features| * |features| * self.prune_ratio)) shuffles.
        """
        worst_score, worst_subset = float('inf'), []
        all_importance_df = None
        time_limit = kwargs.get('time_limit', None)
        num_prune = max(1, int(len(self.prune_candidate_features) * self.prune_ratio))
        resource_per_subset = max(1, num_resource // (len(self.prune_candidate_features)*num_prune))
        performance_gained = True
        while num_resource > 0 and performance_gained:
            performance_gained = False
            candidate_feature_subsets = []
            for feature in [feature for feature in self.prune_candidate_features if feature not in worst_subset]:
                if len(worst_subset) == 0:
                    candidate_feature_subset = feature
                else:
                    curr_subset = worst_subset + [feature]
                    candidate_feature_subset = (f"{self.custom_header}{self.custom_separator}{self.custom_separator.join(curr_subset)}", curr_subset)
                candidate_feature_subsets.append(candidate_feature_subset)
            importance_fn_args = deepcopy(self.importance_fn_args)
            importance_fn_args['features'] = candidate_feature_subsets
            importance_fn_args['time_limit'] = time_limit
            importance_fn_args['num_shuffle_sets'] = resource_per_subset

            time_start = time.time()
            importance_df = self.importance_fn(**importance_fn_args)
            time_elapsed = time.time() - time_start
            if time_limit:
                if time_limit - time_elapsed <= 0:
                    raise TimeLimitExceeded
                else:
                    time_limit = time_limit - time_elapsed

            for custom_feature, info in importance_df.iterrows():
                curr_score = info['importance']
                if curr_score < worst_score:
                    performance_gained = True
                    worst_score = curr_score
                    if self.custom_header in custom_feature:
                        worst_subset = custom_feature.split(self.custom_separator)[1:]
                    else:
                        worst_subset = [custom_feature]

            if all_importance_df is None:
                all_importance_df = importance_df
            else:
                all_importance_df = pd.concat([all_importance_df, importance_df], axis=0)
            num_resource = num_resource - resource_per_subset * len(importance_df)
        return all_importance_df, [all_importance_df]


class FeaturePruneHelper:
    def __init__(self, golden_features: Sequence[str] = [], **kwargs) -> None:
        self.golden_features = golden_features
        self.kept_features = []
        self.pruned_features = []
        self.importance_df = None
        self.custom_header = "AUTOGLUONPRUNE"
        self.custom_separator = "//"

    def set_fi_results(self, importance_df: pd.DataFrame, **kwargs: dict):
        raise NotImplementedError

    def prune_features_from_fi(self, **kwargs: dict) -> Sequence[str]:
        raise NotImplementedError

    def prune_less_features(self, **kwargs: dict) -> Sequence[str]:
        raise NotImplementedError


class SingleFeaturePruneHelper(FeaturePruneHelper):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def set_fi_results(self, importance_df: pd.DataFrame, **kwargs: dict):
        self.importance_df = importance_df.sort_values(by='importance', axis=0)

    def prune_features_from_fi(self, **kwargs: dict) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Given a DataFrame of feature name and importance, return the list of feature names
        whose feature importance score mean was and was not the worst of all individual feature importance
        scores. If feature with worst feature importance is a set of features, remove those features.
        """
        if len(self.kept_features) == 0:
            self.kept_features = self.importance_df.index.tolist()
        if len(self.pruned_features) > 0:
            self.kept_features = self.kept_features + self.pruned_features
            self.pruned_features = []
        worst_feature = self.importance_df.iloc[0].name
        if self.custom_header in worst_feature:
            self.pruned_features = worst_feature.split(self.custom_separator)[1:]
        else:
            self.pruned_features = [worst_feature]
        self.kept_features = list(filter(lambda feature: self.custom_header not in feature and feature not in self.pruned_features, self.kept_features))
        return self.golden_features + self.kept_features, deepcopy(self.pruned_features)

    def prune_features_from_fi_on_performance_loss(self, **kwargs) -> Sequence[str]:
        """
        Return the next best feature subset. Do this buy deleting the feature subset with the lowest
        importance score and running prune_features_from_fi.
        """
        self.importance_df = self.importance_df[1:]
        return self.prune_features_from_fi(**kwargs)


class PercentageFeaturePruneHelper(FeaturePruneHelper):
    def __init__(self, prune_ratio: float = 0.1, threshold: Union[None, float] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prune_ratio = prune_ratio
        self.threshold = threshold

    def set_fi_results(self, importance_df: pd.DataFrame, **kwargs: dict):
        self.importance_df = importance_df.sort_values(by='importance', axis=0).filter(regex=f"^(?!{self.custom_header})", axis=0)

    def prune_features_from_fi(self) -> Tuple[Sequence[str], Sequence[str]]:
        """
        Given a DataFrame of feature name and importance, return the list of feature names
        whose feature importance score mean is not in and in bottom X% of all feature importance scores.
        At least 1 feature will be pruned even if |bottom X%| < 1. Optionally, specify a
        threshold such that features whose feature importance score mean is bottom X% of
        all feature importance scores OR above the given threshold are returned.
        """
        num_prune = max(1, int((len(self.importance_df)+len(self.golden_features)) * self.prune_ratio))
        for index, (name, info) in enumerate(self.importance_df.iterrows()):
            if index >= num_prune or (self.threshold and info['importance'] > self.threshold):
                self.kept_features.append(name)
            else:
                self.pruned_features.append(name)
        return self.golden_features + self.kept_features, deepcopy(self.pruned_features)

    def prune_features_from_fi_on_performance_loss(self, **kwargs) -> Sequence[str]:
        """
        Set self.kept_features and self.pruned_features such that half as many features
        are pruned compared to how things were before calling this method. Assume
        self.pruned_features is always sorted in ascending order (feature importance).
        """
        pruned_features_len = len(self.pruned_features)
        new_pruned_features_len = pruned_features_len//2
        to_remove = self.pruned_features[:new_pruned_features_len]
        to_keep = self.pruned_features[new_pruned_features_len:]
        self.pruned_features = to_remove
        self.kept_features = self.kept_features + to_keep
        return self.golden_features + self.kept_features, deepcopy(self.pruned_features)


class FeatureSelector:
    def __init__(self, fi_helper: FeatureImportanceHelper, fp_helper: FeaturePruneHelper):
        self.fi_helper = fi_helper
        self.fp_helper = fp_helper

    def select_features(self, num_resource: int, **kwargs) -> Tuple[Sequence[str], Sequence[pd.DataFrame]]:
        importance_df, importance_dfs = self.fi_helper.compute_fi(num_resource, **kwargs)
        self.fp_helper.set_fi_results(importance_df=importance_df)
        kept_features, _ = self.fp_helper.prune_features_from_fi()
        return kept_features, importance_dfs

    def select_features_on_performance_loss(self, **kwargs) -> Sequence[str]:
        return self.fp_helper.prune_features_from_fi_on_performance_loss(**kwargs)[0]
