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


def add_noise_column(X: pd.DataFrame, prefix: str, count: int = 1, feature_metadata: FeatureMetadata = None, rng: np.random.Generator = None) -> pd.DataFrame:
    # Create a copy of dataset X with extra synthetic columns generated from standard normal distribution.
    if X is None:
        return None
    X = X.copy()
    if rng is None:
        rng = np.random.default_rng(seed=0)
    for i in range(1, count+1):
        col_name = f"{prefix}_{i}"
        if feature_metadata is not None:
            feature_metadata.type_map_raw[col_name] = R_FLOAT
        noise = rng.standard_normal(len(X))
        X[col_name] = noise
    return X


def merge_importance_dfs(df_old: pd.DataFrame, df_new: pd.DataFrame, iteration_unevaluated: Set[str]) -> pd.DataFrame:
    # Create a dataframe that correctly merges two existing dataframe's permutation feature importance statistics,
    # specifically mean, standard deviation, and shuffle count. For each feature, if one dataframe's mean feature
    # importance is 0, the resulting dataframe will contain the other dataframe's mean feature importance value.
    # df_old is assumed to have been from previous feature importance computation round or even pruning round and
    # can have more features (rows) than df_new.
    if df_old is None:
        return df_new
    assert len(df_old) >= len(df_new), "df_old cannot have less rows than df_new."
    combined_df = df_new.copy()
    for feature in df_new.index.tolist():
        row1, row2 = df_old.loc[feature], df_new.loc[feature]
        mean1, mean2 = row1['importance'], row2['importance']
        stddev1, stddev2 = row1['stddev'], row2['stddev']
        n1, n2 = row1['n'], row2['n']
        if n1 > 0 and n2 > 0 and feature not in iteration_unevaluated:
            combined_df.loc[feature, 'importance'] = (n1*mean1 + n2*mean2) / (n1 + n2)
            combined_df.loc[feature, 'stddev'] = np.sqrt(((n1 - 1) * stddev1 ** 2 + (n2 - 1) * stddev2 ** 2) / (n1 + n2 - 1) +
                                                         (n1 * n2 * (mean1 - mean2) ** 2) / ((n1 + n2) * (n1 + n2 - 1)))
            combined_df.loc[feature, 'n'] = n1 + n2
        elif n2 > 0:
            combined_df.loc[feature, 'importance'] = mean2
            combined_df.loc[feature, 'stddev'] = stddev2
            combined_df.loc[feature, 'n'] = n2
            iteration_unevaluated.remove(feature)
        elif n1 > 0:
            combined_df.loc[feature, 'importance'] = mean1
            combined_df.loc[feature, 'stddev'] = stddev1
            combined_df.loc[feature, 'n'] = n1
        else:
            combined_df.loc[feature, 'importance'] = None
            combined_df.loc[feature, 'stddev'] = None
            combined_df.loc[feature, 'n'] = 0
        combined_df.loc[feature, 'p_value'] = None
    return combined_df.sort_values(by='importance', axis=0, ascending=False)


class FeatureSelector:
    def __init__(self, model: AbstractModel, time_limit: float, seed: int = 0, keep_models: bool = False) -> None:
        self.original_model = model
        self.base_model = model.convert_to_template()
        self.base_model.rename(f'FeatureSelector_{self.base_model.name}')
        if time_limit is None:
            raise AssertionError("Time limit cannot be unspecified.")
        self.original_time_limit = time_limit
        self.time_limit = time_limit
        self.rng = np.random.default_rng(seed)
        self.keep_models = keep_models
        self.is_proxy_model = model.is_valid()
        self.is_bagged = isinstance(model, BaggedEnsembleModel)
        if self.is_proxy_model:
            self.model_fit_time = model.fit_time
            self.model_predict_time = model.predict_time
            self.original_val_score = model.val_score
        else:
            self.model_fit_time = None
            self.model_predict_time = None
            self.original_val_score = None
        # TODO: can we decide how many subsamples to take based on model.predict_time?
        self.trained_models = []
        self.importance_dfs = []
        self.attempted_removals = set()
        self._debug_info = {'exceptions': [], 'index_trajectory': [], 'layer_fit_time': 0., 'total_prune_time': 0., 'total_prune_fit_time': 0.,
                            'total_prune_fi_time': 0., 'score_improvement_from_proxy_yes': 0, 'score_improvement_from_proxy_no': 0, 'kept_ratio': 0.}
        self._fit_time_elapsed = 0.
        self._fi_time_elapsed = 0.
        self.noise_prefix = 'AG_normal_noise'

    def select_features(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, train_subsample_size: int = 50000,
                        fi_subsample_size: int = 5000, prune_ratio: float = 0.05, prune_threshold: float = None, stop_threshold: int = 1,
                        min_fi_samples: int = 10000, max_fi_samples: int = 100000, max_fits: int = 5, improvement_threshold: float = 0.0005,
                        **kwargs) -> Tuple[List[str], AbstractModel]:
        # optionally subsample training data and add noise features to dataset
        original_features = X.columns.tolist()
        auto_threshold = prune_threshold is None
        refit_at_end = auto_threshold and not self.is_proxy_model
        X, y, X_val, y_val, subsampled, kwargs = self.consider_subsampling(X, y, X_val, y_val, train_subsample_size, **kwargs)
        if auto_threshold:
            kwargs['feature_metadata'] = deepcopy(kwargs['feature_metadata']) if 'feature_metadata' in kwargs else None
            X = add_noise_column(X, prefix=self.noise_prefix, feature_metadata=kwargs.get('feature_metadata', None))
            X_val = add_noise_column(X_val, prefix=self.noise_prefix, feature_metadata=kwargs.get('feature_metadata', None))

        candidate_features = X.columns.tolist()
        X_fi, y_fi = (X, y) if self.is_bagged else (X_val, y_val)
        best_info = {'features': candidate_features, 'index': 1, 'model': self.original_model, 'score': round(self.original_val_score, 4)}
        logger.log(30, f"\tPerforming V1 model feature selection with model: {self.base_model.name}, total time limit: {round(self.time_limit, 2)}s, " +
                       f"max fits: {max_fits}, stop threshold: {stop_threshold}, prune ratio: {prune_ratio}, prune threshold: " +
                       f"{'auto' if prune_threshold is None else prune_threshold}.")
        try:
            # fit proxy model once on the subsampled dataset to serve as scoring reference if using subsamples or added a noise column
            index = 1
            if subsampled or auto_threshold:
                model_name = f"{self.base_model.name}_1"
                curr_model, score, _ = self.fit_score_model(deepcopy(self.base_model), X, y, X_val, y_val, candidate_features, model_name, **kwargs)
                best_info['model'], best_info['score'] = curr_model, score
            elif self.keep_models:
                self.trained_models.append(best_info['model'])
            self._debug_info['index_trajectory'].append(True)

            time_budget_fi = max(0.1 * self.model_fit_time, 10 * self.model_predict_time * min(50, len(X.columns)), 60)
            logger.log(30, f"\tExpected model fit time: {round(self.model_fit_time, 2)}s, and expected candidate generation time: {round(time_budget_fi, 2)}s.")
            logger.log(30, f"\tFit 1 ({round(self.model_fit_time, 4)}s): Current score is {best_info['score']}.")
            if self.is_proxy_model and self.time_limit <= self.model_fit_time * (2 if refit_at_end else 1) + time_budget_fi:
                logger.log(30, f"\tInsufficient time to perform even a single pruning round.")
                raise TimeLimitExceeded

            importance_df = None
            for index in range(2, max_fits+1):
                # compute next feature subset to try using feature importance
                model_name = f"{self.base_model.name}_{index}"
                prev_candidate_features = candidate_features
                n_total_fi_samples = max(min_fi_samples, min(max_fi_samples, len(X_fi)))
                prioritize_fi = [feature for feature in best_info['features'] if self.noise_prefix in feature]
                fn_args = {'X': X_fi, 'y': y_fi, 'model': best_info['model'], 'time_budget': time_budget_fi, 'features': best_info['features'],
                           'n_sample': n_total_fi_samples, 'n_subsample': fi_subsample_size, 'prev_importance_df': importance_df, 'prune_ratio': prune_ratio,
                           'prune_threshold': prune_threshold, 'prioritized': prioritize_fi, 'weighted': kwargs.get('weighted', True)}

                candidate_features, importance_df, success, selection_time = self.compute_next_candidate(fn_args, time_budget_fi, prev_candidate_features)
                if not success:
                    logger.log(30, f"\tThere are no more features to prune. Ending...")
                    break
                curr_model, score, fit_time = self.fit_score_model(deepcopy(self.base_model), X, y, X_val, y_val, candidate_features, model_name, **kwargs)
                best_score_multiplier = 1. + improvement_threshold * (1. if score >= 0 else -1.)
                if score > best_info['score'] * best_score_multiplier:
                    logger.log(30, f"\tFit {index} ({fit_time}s): Current score {score} is considerably better than best score {best_info['score']}. Update.")
                    logger.log(30, f"\tOld # Features: {len(best_info['features'])} / New # Features: {len(candidate_features)}.")
                    best_info = {'model': curr_model, 'features': candidate_features, 'score': score, 'index': index}
                    self._debug_info['index_trajectory'].append(True)
                else:
                    logger.log(30, f"\tFit {index} ({fit_time}s): Current score {score} isn't considerably better than best score {best_info['score']}. Retry.")
                    self._debug_info['index_trajectory'].append(False)
                if index - best_info['index'] >= stop_threshold:
                    logger.log(30, f"\tScore has not improved for {stop_threshold} iterations. Ending...")
                    break
                if (refit_at_end and self.time_limit <= 2 * self.model_fit_time + selection_time) or\
                   (not refit_at_end and self.time_limit <= self.model_fit_time + selection_time):
                    logger.log(30, f"\tInsufficient time to finish next pruning round. Ending...")
                    break
        except TimeLimitExceeded:
            logger.log(30, f"\tTime limit exceeded while pruning features. Ending...")
        except Exception as e:
            logger.log(30, f"\tERROR: Exception raised during feature pruning. Reason: {e}. Ending...")
            self._debug_info['exceptions'].append(str(e))

        if auto_threshold:
            best_info['features'] = [feature for feature in best_info['features'] if self.noise_prefix not in feature]
        if refit_at_end:
            model_name = f"{self.base_model.name}_Final" if self.is_proxy_model else self.base_model.name
            model, score, _ = self.fit_score_model(deepcopy(self.base_model), X, y, X_val, y_val, best_info['features'], model_name, **kwargs)
            best_info['model'], best_info['score'] = model, score

        self._debug_info['total_prune_time'] = self.original_time_limit - self.time_limit
        self._debug_info['total_prune_fit_time'] = self._fit_time_elapsed
        self._debug_info['total_prune_fi_time'] = self._fi_time_elapsed
        self._debug_info['kept_ratio'] = len(best_info['features']) / len(original_features)
        logger.log(30, f"\tSuccessfully ended prune loop after {index} iterations. Best score: {best_info['score']}.")
        logger.log(30, f"\tFeature Count: {len(original_features)} -> {len(best_info['features'])} ({round(self.original_time_limit - self.time_limit, 2)}s)")
        return best_info['features'], best_info['model']

    def compute_next_candidate(self, fn_args: dict, time_budget: float, prev_candidate_features: List[str]) -> Tuple[List[str], pd.DataFrame, bool, float]:
        """
        While time allows, repeatedly compute feature importance and generate candidate feature subsets using a fixed time budget.
        If at least one feature can be pruned, return. If no feature is immediately pruned but time remains and some feature's
        importance scores are not calculated, repeat the procedure.
        """
        candidate_features = fn_args['features']
        importance_df = unevaluated_fi_df_template(candidate_features)
        candidate_found = False
        total_feature_selection_time = 0.
        fn_args['iteration_unevaluated'] = set(fn_args['features'])
        while self.time_limit > time_budget + self.model_fit_time:
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
                                     n_sample=10000, n_subsample=5000, prev_importance_df: pd.DataFrame = None, prune_threshold: float = None,
                                     prune_ratio: float = 0.05, prioritized: List[str] = [], iteration_unevaluated: Set[str] = [], weighted: bool = True
                                     ) -> Tuple[List[str], pd.DataFrame, float]:
        """
        Generate promising feature subsets based on permutation feature importance using time as resource. If there isn't time to evaluate
        permutation feature importance for at least n_sample datapoints for all features, only evaluate importance scores for some features.
        Update self.time_limit to account for time taken during feature importance computation.
        """
        # determine how many subsamples and shuffles to use for feature importance calculation
        time_start = time.time()
        n_features = len(features)
        n_subsample = min(n_subsample, len(X))
        n_shuffle = min(np.ceil(n_sample / n_subsample).astype(int), 100)
        auto_threshold = len(prioritized) > 0
        features = self.sort_features_by_priority(features, prioritized, prev_importance_df, iteration_unevaluated)

        # if we do not have enough time to evaluate feature importance for all features, do so only for some (first n_evaluated_features elements of features)
        expected_single_feature_time = 1.1 * self.model_predict_time * (n_subsample / len(X)) * n_shuffle
        n_evaluated_features = max([i for i in range(0, n_features+1) if i * expected_single_feature_time < time_budget])
        if n_evaluated_features == 0:
            return features, unevaluated_fi_df_template(features)
        evaluated_features = features[:n_evaluated_features]
        unevaluated_features = features[n_evaluated_features:]
        logger.log(30, f"\tComputing feature importance for {n_evaluated_features}/{n_features} features with {n_shuffle} shuffles.")
        evaluated_df = model.compute_feature_importance(X=X, y=y, num_shuffle_sets=n_shuffle, subsample_size=n_subsample, features=evaluated_features,
                                                        is_oof=self.is_bagged, silent=True, time_limit=time_budget, random_state=self.rng.integers(0, 1e6))
        if self.is_bagged:
            evaluated_df['n'] = evaluated_df['n'] // len(model.models)

        # if we could not compute feature importance for all features and previous feature importance estimates exist, use them
        importance_df = pd.concat([evaluated_df, unevaluated_fi_df_template(unevaluated_features)])
        importance_df = merge_importance_dfs(df_old=prev_importance_df, df_new=importance_df, iteration_unevaluated=iteration_unevaluated)

        # if auto_threshold, threshold is the mean of noise column importance score
        if auto_threshold:
            noise_rows = importance_df[importance_df.index.isin(prioritized)]
            importance_df = importance_df.drop(prioritized)
            prune_threshold = noise_rows['importance'].mean()

        # use importance_df to generate next candidate features
        time_budget = time_budget - (time.time() - time_start)
        candidate_features = self.prune_features_given_fi(importance_df, prune_threshold, prune_ratio, time_budget, weighted)

        # if noise columns exist, they should never be removed
        if auto_threshold:
            candidate_features = candidate_features + prioritized
            importance_df = pd.concat([importance_df, noise_rows])

        feature_selection_time = time.time() - time_start
        self.time_limit = self.time_limit - feature_selection_time
        self._fi_time_elapsed = self._fi_time_elapsed + feature_selection_time
        return candidate_features, importance_df.sort_values(by='importance', axis=0), feature_selection_time

    def sort_features_by_priority(self, features: List[str], prioritized: List[str], prev_importance_df: pd.DataFrame,
                                  iteration_unevaluated: Set[str]) -> List[str]:
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
            # unevaluated_rows = prev_importance_df[prev_importance_df['importance'].isna()]
            # iteration_unevaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
            #                                                  (prev_importance_df.index.isin(iteration_unevaluated))].sort_values(by='importance')
            # iteration_evaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
            #                                               ~(prev_importance_df.index.isin(iteration_unevaluated_rows.index))].sort_values(by='importance')
            # features = unevaluated_rows.index.tolist() + iteration_unevaluated_rows.index.tolist() + iteration_evaluated_rows.index.tolist()
            unevaluated_rows = prev_importance_df[prev_importance_df['importance'].isna()]
            evaluated_rows = prev_importance_df[~prev_importance_df['importance'].isna()].sort_values(by='importance', ascending=True)
            features = unevaluated_rows.index.tolist() + evaluated_rows.index.tolist()
        if auto_threshold:
            non_prioritized = [feature for feature in features if feature not in prioritized]
            features = prioritized + non_prioritized
        return features

    def prune_features_given_fi(self, importance_df: pd.DataFrame, prune_threshold: float, prune_ratio: float, time_budget: float,
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
        logger.log(30, f"\tNumber of original features above prune threshold {round(prune_threshold, 4)}: {len(above_threshold_rows)}/{len(importance_df)}")
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
            random_state = self.rng.integers(low=0, high=1e6)
            removal_candidate_rows = below_threshold_rows.sample(n=n_remove, random_state=random_state, replace=False, weights=sample_weights)
            removal_candidates = tuple(removal_candidate_rows.index)
            if removal_candidates not in self.attempted_removals:
                acceptance_candidates = importance_df[~importance_df.index.isin(removal_candidates)].index.tolist()
                self.attempted_removals.add(removal_candidates)
                return acceptance_candidates
        return importance_df.index.tolist()

    def fit_score_model(self, model: AbstractModel, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                        features: List[str], model_name: str, **kwargs) -> Tuple[AbstractModel, float, float]:
        """
        Fits and scores a model. Updates self.time_limit. Returns the fitted model, its score, and time elapsed.
        If this is the first time we are fitting a model in the pruning procedure, save time and score statistics.
        """
        time_start = time.time()
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

    def consider_subsampling(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame, train_subsample_size: int,
                             **kwargs) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, bool, dict]:
        """
        If using a proxy model and dataset size is larger than train_sample_size, subsample data to make
        model training faster. If the proxy model is bagged and we have a lot of data, use a non-bagged
        version instead since it is ~10x faster to train. Update fit and predict time estimates accordingly.
        """
        X_train, y_train = X, y
        if self.is_proxy_model and len(X) > train_subsample_size:
            subsampled = True
            X_train = X.sample(train_subsample_size, random_state=0)
            y_train = y.loc[X_train.index]
            replace_bag = kwargs.pop('replace_bag', True)
            if replace_bag and self.is_bagged and len(X) >= train_subsample_size * 2:
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
