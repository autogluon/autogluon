from copy import deepcopy
import logging
import numpy as np
import pandas as pd
import time
from typing import Callable, Sequence, Tuple, Union
from .exceptions import TimeLimitExceeded
# from autogluon.core.models import BaggedEnsembleModel
# from autogluon.tabular.models import CatBoostModel, LGBModel


logger = logging.getLogger(__name__)

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


class ProxyScorer:
    def __init__(self):
        self.y_type = None

    def score_feature(self, x: pd.Series, y: pd.Series, x_type: str, y_type: str, **kwargs) -> dict:
        raise NotImplementedError

    def score_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str, feature_metadata, **kwargs) -> dict:
        result = {}
        if problem_type in ['binary', 'multiclass', 'quantile']:
            self.y_type = 'categorical'
        elif problem_type in ['regression']:
            self.y_type = 'numeric'
        else:
            raise NotImplementedError(f'Proxy Model Scoring not enabled for problem type: {problem_type}.')
        for feature in X.columns:
            feature_type = feature_metadata.get_feature_type_raw(feature)
            if feature_type in ['int', 'category']:
                feature_type = 'categorical'
                result[feature] = self.score_feature(X[feature], y, feature_type, self.y_type, **kwargs)
            elif feature_type in ['float']:
                feature_type = 'numeric'
                result[feature] = self.score_feature(X[feature], y, feature_type, self.y_type, **kwargs)
            else:
                logger.log(10, f"Unknown feature type {feature_type}. Setting score to zero.")
                result[feature] = 0.
        return result


class MutualInformationScorer(ProxyScorer):
    def __init__(self):
        super().__init__()

    def entropy(self, c):
        c_normalized = c / float(np.sum(c))
        c_normalized = c_normalized[np.nonzero(c_normalized)]
        h = -sum(c_normalized * np.log2(c_normalized))
        return h

    def score_feature(self, x: pd.Series, y: pd.Series, x_type: str, y_type: str, **kwargs) -> float:
        if x_type == 'categorical':
            x = x.fillna(x.mode().item())
            x_bin = len(x.unique())
        else:
            x = x.fillna(x.mean())
            x_bin = 'auto'  # int(len(x) * continuous_bin_rate)
        if y_type == 'categorical':
            y = y.fillna(y.mode().item())
            y_bin = len(y.unique())
        else:
            y = y.fillna(y.mean())
            y_bin = 'auto'  # int(len(y) * continuous_bin_rate)
        c_x = np.histogram(x, x_bin)[0]
        c_y = np.histogram(y, y_bin)[0]
        h_x = self.entropy(c_x)
        h_y = self.entropy(c_y)
        c_xy = np.histogram2d(x, y, (len(c_x), len(c_y)))[0]
        h_xy = self.entropy(c_xy)
        mi = h_x + h_y - h_xy
        return mi

"""
class ProxyModelScorer(ProxyScorer):
    def __init__(self, model: str, **kwargs):
        super().__init__()
        if model == 'LGB':
            self.model_type = LGBModel
        else:
            self.model_type = CatBoostModel
        self.num_train_samples = kwargs.get('num_train_samples', 50000)
        self.n_repeats = kwargs.get('n_repeats', 1)
        self.seed = kwargs.get('random_seed', 0)
        # fit_with_prune kwargs
        self.max_num_fit = kwargs.get('max_num_fit', 5)
        self.stop_threshold = kwargs.get('stop_threshold', 3)
        self.prune_ratio = kwargs.get('prune_ratio', 0.1)
        self.num_resource = kwargs.get('num_resource', None)
        self.fi_strategy = kwargs.get('fi_strategy', 'uniform')
        self.fp_strategy = kwargs.get('fp_strategy', 'percentage')
        self.subsample_size = kwargs.get('subsample_size', 5000)
        self.prune_threshold = kwargs.get('prune_threshold', None)
        self.num_min_fi_samples = kwargs.get('num_min_fi_samples', 50000)

    def score_features(self, X: pd.DataFrame, y: pd.Series, problem_type: str, feature_metadata, **kwargs) -> dict:
        # set problem type, evaluation metric, etc to model
        base_model = self.model_type(name='proxy_base_model', problem_type=problem_type, eval_metric=kwargs.get('eval_metric', None))
        self.model = BaggedEnsembleModel(base_model, name='proxy_model', random_state=0)
        indexes = np.random.default_rng(self.seed).choice(len(X), replace=False, size=self.num_train_samples)
        X_train, y_train = X.iloc[indexes], y.iloc[indexes]
        best_model, _ = self.model.fit_with_prune(X=X_train, y=y_train, X_val=None, y_val=None, max_num_fit=self.max_num_fit,
                                                  stop_threshold=self.stop_threshold, prune_ratio=self.prune_ratio, num_resource=self.num_resource,
                                                  fi_strategy=self.fi_strategy, fp_strategy=self.fp_strategy, subsample_size=self.subsample_size,
                                                  prune_threshold=self.prune_threshold, num_min_fi_samples=self.num_min_fi_samples, k_fold=5)
        return
"""
