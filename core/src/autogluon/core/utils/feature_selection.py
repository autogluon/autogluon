from typing import List, Set, Tuple
from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.models.abstract.abstract_model import AbstractModel
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from copy import deepcopy
import logging
from autogluon.core.utils.utils import unevaluated_fi_df_template
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.features.types import R_FLOAT
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)


def add_noise_column(X: pd.DataFrame, prefix: str, rng: np.random.Generator, count: int = 1, feature_metadata: FeatureMetadata = None) -> pd.DataFrame:
    """
    Create a copy of dataset X with extra synthetic columns generated from standard normal distribution.
    """
    if X is None:
        return None
    X = X.copy()
    for i in range(1, count+1):
        col_name = f"{prefix}_{i}"
        if feature_metadata is not None:
            feature_metadata.type_map_raw[col_name] = R_FLOAT
        noise = rng.standard_normal(len(X))
        X[col_name] = noise
    return X


def merge_importance_dfs(df_old: pd.DataFrame, df_new: pd.DataFrame, prev_fit_estimates: Set[str]) -> pd.DataFrame:
    """
    Create a dataframe that correctly merges two existing dataframe's permutation feature importance statistics,
    specifically mean, standard deviation, and shuffle count. For each feature, if one dataframe's feature importance
    has not been calculated, the resulting dataframe will contain the other dataframe's feature importance stats.
    df_old is assumed to have been from previous feature importance computation round or even pruning round and
    can have more features (rows) than df_new. Also, update prev_fit_estimates to indicate the updated feature list that
    uses feature importance values from previous fit.
    """
    if df_old is None:
        return df_new
    assert len(df_old) >= len(df_new), "df_old cannot have less rows than df_new."
    evaluated_old_rows, evaluated_new_rows = df_old[df_old['n'] > 0], df_new[df_new['n'] > 0]
    unevaluated_old_rows, unevaluated_new_rows = df_old[df_old['n'] == 0], df_new[df_new['n'] == 0]
    evaluated_both = evaluated_new_rows.index.intersection(evaluated_old_rows.index).difference(prev_fit_estimates).tolist()
    evaluated_neither = unevaluated_new_rows.index.intersection(unevaluated_old_rows.index).tolist()
    evaluated_old_only = evaluated_old_rows[evaluated_old_rows.index.isin(unevaluated_new_rows.index)].index.tolist()
    evaluated_new_only = evaluated_new_rows[evaluated_new_rows.index.isin(unevaluated_old_rows.index)].index.tolist()
    evaluated_new_first_time = evaluated_new_rows.index.intersection(prev_fit_estimates).tolist()

    # for features with no info on both df_old and df_new, return no info rows
    evaluated_neither_rows = unevaluated_new_rows.loc[evaluated_neither]
    # for features with info on only df_old, return corresponding df_old rows
    evaluated_old_only_rows = evaluated_old_rows.loc[evaluated_old_only]
    # for features with info on only df_new, return corresponding df_new rows
    evaluated_new_only_rows = evaluated_new_rows.loc[evaluated_new_only + evaluated_new_first_time]
    # for features with info on both df_old and df_new, return combined statistics
    evaluated_both_rows = pd.DataFrame()
    evaluated_both_rows_new = evaluated_new_rows.loc[evaluated_both].sort_index()
    evaluated_both_rows_old = evaluated_old_rows.loc[evaluated_both].sort_index()
    mean_old, mean_new = evaluated_both_rows_old['importance'], evaluated_both_rows_new['importance']
    stddev_old, stddev_new = evaluated_both_rows_old['stddev'], evaluated_both_rows_new['stddev']
    n_old, n_new = evaluated_both_rows_old['n'], evaluated_both_rows_new['n']
    evaluated_both_rows['importance'] = (n_old * mean_old + n_new * mean_new) / (n_old + n_new)
    evaluated_both_rows['stddev'] = (((n_old - 1) * stddev_old ** 2 + (n_new - 1) * stddev_new ** 2) / (n_old + n_new - 1) +
                                     (n_old * n_new * (mean_old - mean_new) ** 2) / ((n_old + n_new) * (n_old + n_new - 1))) ** 0.5
    evaluated_both_rows['p_value'] = None
    evaluated_both_rows['n'] = n_old + n_new
    # remove features evaluated in df_new from prev_fit_estimates if they exist
    prev_fit_estimates.difference_update(evaluated_new_rows.index.tolist())
    return pd.concat([evaluated_both_rows, evaluated_new_only_rows, evaluated_old_only_rows, evaluated_neither_rows])


def compute_expected_fi_time_single(X_fi: pd.DataFrame, model_predict_time: float, n_subsample: int,
                                    n_total_sample: int, safety_time_multiplier: float = 1.) -> float:
    """
    Return the expected time to compute permutation feature importance for a single feature on n_total_sample datapoints.
    """
    n_subsample = min(n_subsample, len(X_fi))
    n_shuffle = min(np.ceil(n_total_sample / n_subsample).astype(int), 100)
    expected_single_feature_time = safety_time_multiplier * model_predict_time * (n_subsample / len(X_fi)) * n_shuffle
    return expected_single_feature_time


class FeatureSelector:
    def __init__(self, model: AbstractModel, time_limit: float, seed: int = 0, keep_models: bool = False) -> None:
        assert time_limit is not None, "Time limit cannot be unspecified."
        assert model.is_valid(), "Model must have been fit."
        self.original_model = model
        self.base_model = model.convert_to_template()
        self.base_model.rename(f'FeatureSelector_{self.base_model.name}')
        self.original_time_limit = time_limit
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        self.keep_models = keep_models
        self.is_bagged = isinstance(model, BaggedEnsembleModel)
        self.model_fit_time = model.fit_time
        self.model_predict_time = model.predict_time
        self.original_val_score = model.val_score
        # TODO: can we decide how many subsamples to take based on model.predict_time?
        self.trained_models = []
        self.importance_dfs = []
        self.attempted_removals = set()
        self.noise_prefix = 'AG_normal_noise'
        self.safety_time_multiplier = 1.1
        self._debug_info = {'exceptions': [], 'index_trajectory': [], 'layer_fit_time': 0., 'total_prune_time': 0., 'total_prune_fit_time': 0.,
                            'total_prune_fi_time': 0., 'score_improvement_from_proxy_yes': 0, 'score_improvement_from_proxy_no': 0, 'kept_ratio': 0.}
        self._fit_time_elapsed = 0.
        self._fi_time_elapsed = 0.

    def select_features(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, train_subsample_size: int = 50000,
                        fi_subsample_size: int = 5000, prune_ratio: float = 0.05, prune_threshold: float = None, stop_threshold: int = 1,
                        min_fi_samples: int = 10000, max_fi_samples: int = 100000, **kwargs) -> Tuple[List[str], List[pd.DataFrame]]:

        X, y, X_val, y_val, subsampled, kwargs = self.consider_subsampling(X, y, X_val, y_val, train_subsample_size, min_fi_samples, **kwargs)
        original_features = X.columns.tolist()
        auto_threshold = prune_threshold is None
        if auto_threshold:
            kwargs['feature_metadata'] = deepcopy(kwargs['feature_metadata']) if 'feature_metadata' in kwargs else None
            X = add_noise_column(X, self.noise_prefix, self.rng, feature_metadata=kwargs.get('feature_metadata', None))
            X_val = add_noise_column(X_val, self.noise_prefix, self.rng, feature_metadata=kwargs.get('feature_metadata', None))
        X_fi, y_fi = (X, y) if self.is_bagged else (X_val, y_val)
        if self.is_bagged and self.original_model._child_oof and self.model_fit_time is not None:
            self.model_fit_time = self.model_fit_time * kwargs['k_fold']

        logger.log(30, f"\tPerforming V2 model feature selection with model: {self.base_model.name}, total time limit: {round(self.time_limit, 2)}s, " +
                       f"stop threshold: {stop_threshold}, prune ratio: {prune_ratio}, prune threshold: {'auto' if not prune_threshold else prune_threshold}.")

        try:
            index = 1
            candidate_features = X.columns.tolist()
            best_info = {'features': candidate_features, 'index': 1, 'model': self.original_model, 'score': round(self.original_val_score, 4)}
            if self.model_fit_time is not None and self.time_limit < self.model_fit_time:
                logger.log(30, f"\tInsufficient time to perform even a single pruning round.")
                raise TimeLimitExceeded

            # fit proxy model once on the subsampled dataset to serve as scoring reference if using subsamples or added a noise column
            if subsampled or auto_threshold or (self.is_bagged and self.original_model._child_oof):
                best_info['model'], best_info['score'], _ = self.fit_score_model(X, y, X_val, y_val, candidate_features, f"{self.base_model.name}_1", **kwargs)
            self._debug_info['index_trajectory'].append(True)

            n_total_fi_samples = max(min_fi_samples, min(max_fi_samples, len(X_fi)))
            fi_time_single = compute_expected_fi_time_single(X_fi, self.model_predict_time, fi_subsample_size, n_total_fi_samples, self.safety_time_multiplier)
            time_budget_fi = min(fi_time_single * min(len(original_features), 50), 300)

            logger.log(30, f"\tExpected model fit time: {round(self.model_fit_time, 2)}s, and expected candidate generation time: {round(time_budget_fi, 2)}s.")
            logger.log(30, f"\tFit 1 ({round(self.model_fit_time, 2)}s): Current score is {best_info['score']}.")
            if self.time_limit < self.model_fit_time + time_budget_fi:
                logger.log(30, f"\tInsufficient time to perform even a single pruning round.")
                raise TimeLimitExceeded

            importance_df = None
            while True:
                index = index + 1
                model_name = f"{self.base_model.name}_{index}"
                prev_candidate_features = candidate_features
                prioritize_fi = set([feature for feature in best_info['features'] if self.noise_prefix in feature])
                fn_args = {'X': X_fi, 'y': y_fi, 'model': best_info['model'], 'time_budget': time_budget_fi, 'features': best_info['features'],
                           'n_sample': n_total_fi_samples, 'n_subsample': fi_subsample_size, 'prev_importance_df': importance_df, 'prune_ratio': prune_ratio,
                           'prune_threshold': prune_threshold, 'prioritized': prioritize_fi, 'weighted': kwargs.get('weighted', True)}
                candidate_features, importance_df, success, selection_time = self.compute_next_candidate(fn_args, time_budget_fi, prev_candidate_features)
                if not success:
                    logger.log(30, f"\tThere are no more features to prune. Ending...")
                    break
                curr_model, score, fit_time = self.fit_score_model(X, y, X_val, y_val, candidate_features, model_name, **kwargs)

                if score > best_info['score']:
                    logger.log(30, f"\tFit {index} ({fit_time}s): Current score {score} is better than best score {best_info['score']}. Updating model.")
                    logger.log(30, f"\tOld # Features: {len(best_info['features'])} / New # Features: {len(candidate_features)}.")
                    best_info = {'model': curr_model, 'features': candidate_features, 'score': score, 'index': index}
                    self._debug_info['index_trajectory'].append(True)
                else:
                    logger.log(30, f"\tFit {index} ({fit_time}s): Current score {score} is not better than best score {best_info['score']}. Retrying.")
                    self._debug_info['index_trajectory'].append(False)

                if index - best_info['index'] >= stop_threshold:
                    logger.log(30, f"\tScore has not improved for {stop_threshold} iterations. Ending...")
                    break
                if self.time_limit <= self.model_fit_time + selection_time:
                    logger.log(30, f"\tInsufficient time to finish next pruning round. Ending...")
                    break
        except TimeLimitExceeded:
            logger.log(30, f"\tTime limit exceeded while pruning features. Ending...")
        except Exception as e:
            logger.log(30, f"\tERROR: Exception raised during feature pruning. Reason: {e}. Ending...")
            self._debug_info['exceptions'].append(str(e))

        if auto_threshold:
            best_info['features'] = [feature for feature in best_info['features'] if self.noise_prefix not in feature]

        self._debug_info['total_prune_time'] = self.original_time_limit - self.time_limit
        self._debug_info['total_prune_fit_time'] = self._fit_time_elapsed
        self._debug_info['total_prune_fi_time'] = self._fi_time_elapsed
        self._debug_info['kept_ratio'] = len(best_info['features']) / len(original_features)
        logger.log(30, f"\tSuccessfully ended prune loop after {index} iterations. Best score: {best_info['score']}.")
        logger.log(30, f"\tFeature Count: {len(original_features)} -> {len(best_info['features'])} ({round(self.original_time_limit - self.time_limit, 2)}s)")
        return best_info['features'], best_info['model']

    def compute_next_candidate(self, fn_args: dict, round_time_budget: float, prev_candidate_features: List[str]) -> Tuple[List[str], pd.DataFrame, bool, float]:
        """
        While time allows, repeatedly compute feature importance and generate candidate feature subsets using a fixed time budget.
        If at least one feature can be pruned, return. If no feature is immediately pruned but time remains and some feature's
        importance scores are not calculated, repeat the procedure.
        """
        candidate_features = fn_args['features']
        importance_df = unevaluated_fi_df_template(candidate_features)
        candidate_found = False
        total_feature_selection_time = 0.
        if fn_args.get('prev_importance_df', None) is not None:
            prev_importance_df = fn_args['prev_importance_df']
            fn_args['prev_fit_estimates'] = set(prev_importance_df[prev_importance_df['n'] > 0].index.tolist())
        while self.time_limit > round_time_budget + self.model_fit_time:
            candidate_features, importance_df, feature_selection_time = self.compute_next_candidate_round(**fn_args)
            # HACK: Line below is needed to get this working with n-repeated bagged models. Related to feature ordering.
            candidate_features = [feature for feature in fn_args['X'].columns.tolist() if feature in candidate_features]
            total_feature_selection_time = total_feature_selection_time + feature_selection_time
            candidate_set, best_set, prev_candidate_set = set(candidate_features), set(fn_args['features']), set(prev_candidate_features)
            candidate_found = candidate_set != best_set and candidate_set != prev_candidate_set and len(candidate_set) > 0
            all_features_evaluated = len(importance_df[importance_df['importance'].isna()]) == 0
            fn_args['prev_importance_df'] = importance_df
            if candidate_found or all_features_evaluated:
                break
        logger.log(30, f"\tCandidate generation time: ({round(total_feature_selection_time, 2)}s), Cardinality: {len(candidate_features)}")
        return candidate_features, importance_df, candidate_found, total_feature_selection_time

    def compute_next_candidate_round(self, X: pd.DataFrame, y: pd.Series, model: AbstractModel, time_budget: float, features: List[str],
                                     n_sample: int = 10000, n_subsample: int = 5000, prev_importance_df: pd.DataFrame = None, prune_threshold: float = None,
                                     prune_ratio: float = 0.05, prioritized: Set[str] = set(), prev_fit_estimates: Set[str] = set(),
                                     weighted: bool = True) -> Tuple[List[str], pd.DataFrame, float]:
        """
        Compute permutation feature importance for as many features as possible under time_budget. Ensure each returned feature importance
        numbers are evaluated across at least n_sample datapoints.
        """
        # determine how many subsamples and shuffles to use for feature importance calculation
        time_start = time.time()
        n_features = len(features)
        n_subsample = min(n_subsample, len(X))
        n_shuffle = min(np.ceil(n_sample / n_subsample).astype(int), 100)
        expected_single_feature_time = compute_expected_fi_time_single(X, self.model_predict_time, n_subsample, n_sample, self.safety_time_multiplier)
        auto_threshold = len(prioritized) > 0
        features = self.sort_features_by_priority(features, prioritized, prev_importance_df, prev_fit_estimates)

        # if we do not have enough time to evaluate feature importance for all features, do so only for some (first n_evaluated_features elements of features)
        n_evaluated_features = max([i for i in range(0, n_features+1) if i * expected_single_feature_time <= time_budget])
        if n_evaluated_features == 0:
            return features, unevaluated_fi_df_template(features)
        evaluated_features = features[:n_evaluated_features]
        unevaluated_features = features[n_evaluated_features:]
        time_budget_fi = time_budget - (time.time() - time_start)
        logger.log(30, f"\tComputing feature importance for {n_evaluated_features}/{n_features} features with {n_shuffle} shuffles.")
        fi_kwargs = {'X': X, 'y': y, 'num_shuffle_sets': n_shuffle, 'subsample_size': n_subsample, 'features': evaluated_features,
                     'time_limit': time_budget_fi, 'silent': True}
        fi_kwargs.update({'is_oof': True} if self.is_bagged else {})
        evaluated_df = model.compute_feature_importance(**fi_kwargs)
        if self.is_bagged:
            evaluated_df['n'] = evaluated_df['n'] // len(model.models)

        # if we could not compute feature importance for all features and previous feature importance estimates exist, use them
        importance_df = pd.concat([evaluated_df, unevaluated_fi_df_template(unevaluated_features)])
        importance_df = merge_importance_dfs(prev_importance_df, importance_df, prev_fit_estimates)

        # if auto_threshold, threshold is the mean of noise column importance score
        if auto_threshold:
            noise_rows = importance_df[importance_df.index.isin(prioritized)]
            importance_df = importance_df.drop(prioritized)
            prune_threshold = noise_rows['importance'].mean()

        # use importance_df to generate next candidate features
        time_budget_select = time_budget - (time.time() - time_start)
        candidate_features = self.compute_next_candidate_given_fi(importance_df, prune_threshold, prune_ratio, time_budget_select, weighted)

        # if noise columns exist, they should never be removed
        if auto_threshold:
            candidate_features = candidate_features + list(prioritized)
            importance_df = pd.concat([importance_df, noise_rows])

        feature_selection_time = time.time() - time_start
        self.time_limit = self.time_limit - feature_selection_time
        self._fi_time_elapsed = self._fi_time_elapsed + feature_selection_time
        return candidate_features, importance_df.sort_values(by='importance', axis=0), feature_selection_time

    def sort_features_by_priority(self, features: List[str], prioritized: Set[str], prev_importance_df: pd.DataFrame, prev_fit_estimates: Set[str]) -> List[str]:
        """
        Return a list of features sorted by feature importance calculation priority in ascending order.
        If prev_importance_df does not exist and not using auto_threshold, return the original list.
        If prev_importance_df exists, features whose importance scores have not been calculated are
        prioritized first followed by features with lowest previous importance scores estimates. If using
        auto_threshold mode, noise columns are prioritized first since their scores are needed to determine
        the pruning threshold.
        """
        is_first_run = prev_importance_df is None
        auto_threshold = len(prioritized) > 0
        if not is_first_run:
            prev_deleted_features = [feature for feature in prev_importance_df.index if feature not in features]
            prev_importance_df = prev_importance_df.drop(prev_deleted_features)
            unevaluated_rows = prev_importance_df[prev_importance_df['importance'].isna()]
            prev_fit_evaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
                                                          (prev_importance_df.index.isin(prev_fit_estimates))].sort_values(by='importance')
            curr_fit_evaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
                                                         ~(prev_importance_df.index.isin(prev_fit_estimates))].sort_values(by='importance')
            features = unevaluated_rows.index.tolist() + prev_fit_evaluated_rows.index.tolist() + curr_fit_evaluated_rows.index.tolist()
        if auto_threshold:
            features = list(prioritized) + [feature for feature in features if feature not in prioritized]
        return features

    def compute_next_candidate_given_fi(self, importance_df: pd.DataFrame, prune_threshold: float, prune_ratio: float, time_budget: float,
                                        weighted: bool = True) -> List[str]:
        """
        Keep features whose importance scores are above threshold or have not yet had a chance to be calculated,
        as well as some features whose importance scores are below threshold if more than prune_ratio * num features
        features are below threshold. In the latter case, randomly sample without replacement from features whose
        importance scores are below threshold until removal candidate configuration that has not yet been tried
        is encountered. Give higher probability to features whose scores are lower than others when sampling.
        """
        time_start = time.time()
        n_remove = max(1, int(prune_ratio * len(importance_df)))
        above_threshold_rows = importance_df[(importance_df['importance'] > prune_threshold) | (importance_df['importance'].isna())]
        below_threshold_rows = importance_df[importance_df['importance'] <= prune_threshold].sort_values(by='importance', axis=0, ascending=True)
        logger.log(30, f"\tNumber of identified features below prune threshold {round(prune_threshold, 4)}: {len(below_threshold_rows)}/{len(importance_df)}")
        if len(below_threshold_rows) <= n_remove:
            acceptance_candidates = above_threshold_rows.index.tolist()
            self.attempted_removals.add(tuple(below_threshold_rows.index))
            return acceptance_candidates

        # Try removing features with lowest importance first
        removal_candidate_rows = below_threshold_rows[:n_remove]
        removal_candidates = tuple(removal_candidate_rows.index)
        if removal_candidates not in self.attempted_removals:
            acceptance_candidates = importance_df[~importance_df.index.isin(removal_candidates)].index.tolist()
            self.attempted_removals.add(removal_candidates)
            return acceptance_candidates

        sample_weights = [1/i for i in range(1, len(below_threshold_rows)+1)] if weighted else None
        while time_budget - (time.time() - time_start) > 0:
            random_state = self.rng.integers(low=0, high=1e5)
            removal_candidate_rows = below_threshold_rows.sample(n=n_remove, random_state=random_state, replace=False, weights=sample_weights)
            removal_candidates = tuple(removal_candidate_rows.index)
            if removal_candidates not in self.attempted_removals:
                acceptance_candidates = importance_df[~importance_df.index.isin(removal_candidates)].index.tolist()
                self.attempted_removals.add(removal_candidates)
                return acceptance_candidates
        return importance_df.index.tolist()

    def fit_score_model(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                        features: List[str], model_name: str, **kwargs) -> Tuple[AbstractModel, float, float]:
        """
        Fits and scores a model. Updates self.time_limit. Returns the fitted model, its score, and time elapsed.
        If this is the first time we are fitting a model in the pruning procedure, save time and score statistics.
        """
        time_start = time.time()
        model = self.base_model.convert_to_template()
        X = X[features]
        X_val = None if self.is_bagged else X_val[features]
        model.rename(model_name)
        if self.is_bagged:
            kwargs['use_child_oof'] = False
        if 'time_limit' in kwargs:
            kwargs['time_limit'] = self.time_limit
        model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)
        fit_time = time.time() - time_start
        time_start = time.time()
        score = model.score_with_oof(y) if self.is_bagged else model.score(X=X_val, y=y_val)
        predict_time = time.time() - time_start
        self.time_limit = self.time_limit - (fit_time + predict_time)
        if self.model_fit_time is None:
            self.model_fit_time = fit_time
        if self.model_predict_time is None:
            self.model_predict_time = predict_time
        if self.original_val_score is None:
            self.original_val_score = score
        if self.keep_models:
            self.trained_models.append(model)
        self._fit_time_elapsed = self._fit_time_elapsed + (fit_time + predict_time)
        return model, round(score, 4), round(fit_time + predict_time, 4)

    def consider_subsampling(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame, train_subsample_size: int, min_fi_samples: int,
                             **kwargs) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, bool, dict]:
        """
        If using a proxy model and dataset size is larger than train_sample_size, subsample data to make
        model training faster. If the proxy model is bagged and we have a lot of data, use a non-bagged
        version instead since it is ~10x faster to train. Update fit and predict time estimates accordingly.
        """
        X_train, y_train = X, y
        if len(X) > train_subsample_size:
            subsampled = True
            X_train = X.sample(train_subsample_size, random_state=self.rng.integers(low=0, high=1e5))
            y_train = y.loc[X_train.index]
            if kwargs.pop('replace_bag', True) and self.is_bagged and len(X) >= train_subsample_size + min_fi_samples:
                self.is_bagged = False
                original_k_fold = kwargs['k_fold']
                self.base_model = self.original_model.convert_to_template_child()
                self.base_model.rename(self.original_model.name.replace('_BAG', ''))
                X_val = X[~X.index.isin(X_train.index)]
                y_val = y.loc[X_val.index]
                if self.model_fit_time is not None:
                    self.model_fit_time = self.model_fit_time / original_k_fold
                if self.model_predict_time is not None:
                    self.model_predict_time = self.model_predict_time / original_k_fold
        else:
            subsampled = False
        return X_train, y_train, X_val, y_val, subsampled, kwargs
