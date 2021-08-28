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
import traceback

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


def merge_importance_dfs(df_old: pd.DataFrame, df_new: pd.DataFrame, using_prev_fit_fi: Set[str]) -> pd.DataFrame:
    """
    Create a dataframe that correctly merges two existing dataframe's permutation feature importance statistics,
    specifically mean, standard deviation, and shuffle count. For each feature, if one dataframe's feature importance
    has not been calculated, the resulting dataframe will contain the other dataframe's feature importance stats.
    df_old is assumed to have been from previous feature importance computation round or even pruning round and
    can have more features (rows) than df_new. Also, update using_prev_fit_fi to indicate the updated feature list that
    uses feature importance values from previous fit.
    """
    if df_old is None:
        # Remove features whose importance has just been computed from using_prev_fit_fi if they exist
        using_prev_fit_fi.difference_update(df_new[df_new['n'] > 0].index.tolist())
        return df_new
    assert len(df_old) >= len(df_new), "df_old cannot have less rows than df_new."
    evaluated_old_rows, evaluated_new_rows = df_old[df_old['n'] > 0], df_new[df_new['n'] > 0]
    unevaluated_old_rows, unevaluated_new_rows = df_old[df_old['n'] == 0], df_new[df_new['n'] == 0]
    evaluated_both = evaluated_new_rows.index.intersection(evaluated_old_rows.index).difference(using_prev_fit_fi).tolist()
    evaluated_neither = unevaluated_new_rows.index.intersection(unevaluated_old_rows.index).tolist()
    evaluated_old_only = evaluated_old_rows[evaluated_old_rows.index.isin(unevaluated_new_rows.index)].index.tolist()
    evaluated_new_only = evaluated_new_rows[evaluated_new_rows.index.isin(unevaluated_old_rows.index)].index.tolist()
    evaluated_new_first_time = evaluated_new_rows.index.intersection(using_prev_fit_fi).tolist()

    # for features with no info on both df_old and df_new, return no info rows
    evaluated_neither_rows = unevaluated_new_rows.loc[evaluated_neither]
    # for features with info on only df_old, return corresponding df_old rows
    evaluated_old_only_rows = evaluated_old_rows.loc[evaluated_old_only]
    # for features with info on only df_new or whose df_old feature importance came from the previous model, return corresponding df_new rows
    evaluated_new_only_rows = evaluated_new_rows.loc[evaluated_new_only + evaluated_new_first_time]
    # for features with info on both df_new and whose df_old feature importance came from the current model, return combined statistics
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
    # remove features evaluated in df_new from using_prev_fit_fi if they exist
    using_prev_fit_fi.difference_update(evaluated_new_rows.index.tolist())
    result = pd.concat([evaluated_both_rows, evaluated_new_only_rows, evaluated_old_only_rows, evaluated_neither_rows]).sort_values('importance')
    assert len(result) == len(df_new), "Length of the updated DataFrame must be equal to the inputted DataFrame."
    return result


def sort_features_by_priority(features: List[str], prioritized: Set[str], prev_importance_df: pd.DataFrame, using_prev_fit_fi: Set[str]) -> List[str]:
    """
    Return a list of features sorted by feature importance calculation priority in ascending order.
    If prev_importance_df does not exist and not using auto_threshold, return the original list.
    If prev_importance_df exists, features whose importance scores have never been calculated are
    prioritized first followed by features whose importance scores are from previous fitted models and
    lastly the rest of the features sorted by previous importance scores estimates in ascending order. If using
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
                                                      (prev_importance_df.index.isin(using_prev_fit_fi))].sort_values(by='importance')
        curr_fit_evaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
                                                     ~(prev_importance_df.index.isin(using_prev_fit_fi))].sort_values(by='importance')
        features = unevaluated_rows.index.tolist() + prev_fit_evaluated_rows.index.tolist() + curr_fit_evaluated_rows.index.tolist()
    if auto_threshold:
        features = list(prioritized) + [feature for feature in features if feature not in prioritized]
    return features


class FeatureSelector:
    def __init__(self, model: AbstractModel, time_limit: float, seed: int = 0, keep_models: bool = False, raise_exception=False) -> None:
        # TODO: Make this work with unlabelled data
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
        self.trained_models = []
        self.importance_dfs = []
        self.attempted_removals = set()
        self.noise_prefix = 'AG_normal_noise'
        self.safety_time_multiplier = 1.1
        self.max_n_shuffle = 100
        self.max_time_budget_fi = 300
        self.raise_exception = raise_exception
        self._debug_info = {'exceptions': [], 'index_trajectory': [], 'layer_fit_time': 0., 'total_prune_time': 0., 'total_prune_fit_time': 0.,
                            'total_prune_fi_time': 0., 'score_improvement_from_proxy_yes': 0, 'score_improvement_from_proxy_no': 0, 'kept_ratio': 0.}
        self._fit_time_elapsed = 0.
        self._fi_time_elapsed = 0.

    def select_features(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, n_train_subsample: int = 50000,
                        n_fi_subsample: int = 5000, prune_threshold: float = None, prune_ratio: float = 0.05, stop_threshold: int = 10,
                        improvement_threshold: float = 1e-6, max_fits: int = None, **kwargs) -> List[str]:
        """
        Performs time-aware recursive feature elimination based on permutation feature importance. While time remains, compute feature importance
        score for as many features as possible over at least min_fi_samples validation datapoints, discard features whose score is less than or
        equal to prune_threshold, fit the model on those features and keep the feature subset if its validation score is better, and repeat.

        Parameters
        ----------
        X : pd.DataFrame
            Training data features
        y : pd.Series
            Training data labels
        X_val: pd.DataFrame
            Validation data features. Can be None.
        y_val: pd.Series
            Validation data labels. Can be None.
        n_train_subsample : int, default 50000
            If the training dataset has more than this amount of datapoints, sample this many datapoints without replacement and use
            them as proxy model training data.
        n_fi_subsample : int, default 5000
            Sample this many datapoints and shuffle them when computing permutation feature importance. If this number is higher than
            the number of feature importance datapoints, set n_fi_subsample = number of feature importance datapoints.
        prune_threshold : float, default None
            Consider features whose feature importance scores are below this threshold for pruning. If set to None, a synthetic columns
            whose values come from standard normal distribution will be appended to the original dataset, and any features whose feature
            importance score is lower than the synthetic feature's score will be considered for pruning.
        prune_ratio : float, default 0.05
            Prune up to prune_ratio * number of current features at once whose feature importance scores are below prune_threshold when
            generating new feature subset candidates. At least one feature is always removed if there are any feature whose importance
            score is below prune_threshold.
        stop_threshold : int, default 10
            If the validation scores of models fit on pruned data do not improve for stop_threshold iterations, end the pruning procedure.
        improvement_threshold: int, default = 1e-6
            The newly fitted model's validation score must be >= improvement_threshold * best validation score seen so far for its input
            feature subset to be considered to be superior to the previous feature subset.
        max_fits: int, default None
            If this many models have been fitted during feature pruning, exit the feature pruning loop. Can potentially prevent overfitting.

        Returns
        -------
        candidate_features : List[str]
            Optimal feature subset selected by this method. Set to original features if no features are below pruning threshold or we run
            out of time before finding a better feature subset.
        """
        logger.log(30, f"\tPerforming V3 feature selection with model: {self.base_model.name}, total time limit: {round(self.time_limit, 2)}s, " +
                       f"stop threshold: {stop_threshold}, prune ratio: {prune_ratio}, prune threshold: {'auto' if not prune_threshold else prune_threshold}.")
        original_features = X.columns.tolist()
        X, y, X_val, y_val, X_fi, y_fi, auto_threshold, subsampled, kwargs = self.setup(X, y, X_val, y_val, n_train_subsample, prune_threshold, kwargs)
        try:
            index = 1
            candidate_features = X.columns.tolist()
            best_info = {'features': candidate_features, 'index': 1, 'model': self.original_model, 'score': round(self.original_val_score, 4)}
            if self.model_fit_time is not None and self.time_limit < self.model_fit_time:
                logger.log(30, f"\tNo time to perform the next pruning round (remaining: {self.time_limit}, needed: {self.model_fit_time}).")
                raise TimeLimitExceeded

            # fit proxy model once on the subsampled dataset to serve as scoring reference if using subsamples or added a noise column
            if subsampled or auto_threshold or (self.is_bagged and self.original_model._child_oof):
                best_info['model'], best_info['score'], _ = self.fit_score_model(X, y, X_val, y_val, candidate_features, f"{self.base_model.name}_1", **kwargs)
            self._debug_info['index_trajectory'].append(True)

            time_budget_fi = self.compute_time_budget_fi(X_fi=X_fi, n_subsample=n_fi_subsample, **kwargs)
            logger.log(30, f"\tExpected model fit time: {round(self.model_fit_time, 4)}s, and expected candidate generation time: {round(time_budget_fi, 2)}s.")
            logger.log(30, f"\tFit 1 ({round(self.model_fit_time, 4)}s): Current score is {best_info['score']}.")
            if self.time_limit < self.model_fit_time + time_budget_fi:
                logger.log(30, f"\tNo time to perform the next pruning round (remaining: {self.time_limit}, needed: {self.model_fit_time + time_budget_fi}).")
                self._debug_info['exceptions'].append(f"exiting after 1 model fit (remaining: {self.time_limit}, needed: {self.model_fit_time+time_budget_fi})")
                raise TimeLimitExceeded

            importance_df = None
            while True:
                index = index + 1
                model_name = f"{self.base_model.name}_{index}"
                prev_candidate_features = candidate_features
                prioritize_fi = set([feature for feature in best_info['features'] if self.noise_prefix in feature])
                fn_args = {'X': X_fi, 'y': y_fi, 'model': best_info['model'], 'time_budget': time_budget_fi, 'features': best_info['features'],
                           'n_subsample': n_fi_subsample, 'prune_threshold': prune_threshold, 'prune_ratio': prune_ratio, 'prioritized': prioritize_fi}
                fn_args.update(self.get_extra_fn_args(**kwargs))
                candidate_features, importance_df, success, prune_time = self.compute_next_candidate(fn_args, time_budget_fi, prev_candidate_features,
                                                                                                     importance_df)
                if not success:
                    logger.log(30, f"\tTime is up while computing feature importance or there are no more features to prune. Ending...")
                    break
                curr_model, score, fit_time = self.fit_score_model(X, y, X_val, y_val, candidate_features, model_name, **kwargs)

                if score >= best_info['score'] * (1 + improvement_threshold):
                    logger.log(30, f"\tFit {index} ({fit_time}s): Current score {score} is better than best score {best_info['score']}. Updating model.")
                    logger.log(30, f"\tOld # Features: {len(best_info['features'])} / New # Features: {len(candidate_features)}.")
                    prev_best_model = best_info['model']
                    best_info = {'model': curr_model, 'features': candidate_features, 'score': score, 'index': index}
                    if not self.keep_models:
                        prev_best_model.delete_from_disk()
                    self._debug_info['index_trajectory'].append(True)
                else:
                    logger.log(30, f"\tFit {index} ({fit_time}s): Current score {score} is not better than best score {best_info['score']}. Retrying.")
                    if not self.keep_models:
                        curr_model.delete_from_disk()
                    self._debug_info['index_trajectory'].append(False)

                if max_fits is not None and index >= max_fits:
                    logger.log(30, f"\tReached maximum number of fits {max_fits}. Ending...")
                    break
                if index - best_info['index'] >= stop_threshold:
                    logger.log(30, f"\tScore has not improved for {stop_threshold} iterations. Ending...")
                    break
                if self.time_limit <= self.model_fit_time + prune_time:
                    logger.log(30, f"\tInsufficient time to finish next pruning round. Ending...")
                    break
        except TimeLimitExceeded:
            logger.log(30, f"\tTime limit exceeded while pruning features. Ending...")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"\tERROR: Exception raised during feature pruning. Reason: {e}. Ending...")
            self._debug_info['exceptions'].append(str(e))
            if self.raise_exception:
                raise e

        if auto_threshold:
            best_info['features'] = [feature for feature in best_info['features'] if self.noise_prefix not in feature]
        if not self.keep_models:
            best_info['model'].delete_from_disk()
        self._debug_info['total_prune_time'] = self.original_time_limit - self.time_limit
        self._debug_info['total_prune_fit_time'] = self._fit_time_elapsed
        self._debug_info['total_prune_fi_time'] = self._fi_time_elapsed
        self._debug_info['kept_ratio'] = len(best_info['features']) / len(original_features)
        logger.log(30, f"\tSuccessfully ended prune loop after {index} iterations. Best score: {best_info['score']}.")
        logger.log(30, f"\tFeature Count: {len(original_features)} -> {len(best_info['features'])} ({round(self.original_time_limit - self.time_limit, 2)}s)")
        return best_info['features']

    def compute_next_candidate(self, fn_args: dict, round_time_budget: float, prev_candidate_features: List[str], prev_importance_df: pd.DataFrame,
                               ) -> Tuple[List[str], pd.DataFrame, bool, float]:
        """
        While time allows, repeatedly compute feature importance and generate candidate feature subsets using a fixed time budget.
        If at least one feature can be pruned, return. If no feature is immediately pruned but time remains and some feature's
        importance scores are not calculated, repeat the procedure. Note: If we have feature importance information from previously
        fitted model, make use of them since we might not have time this iteration to evaluate importance scores for all features.
        Any feature importance calculated within this call of compute_next_candidate will override previous feature importance info.
        """
        candidate_features = prev_candidate_features
        importance_df = unevaluated_fi_df_template(candidate_features)
        candidate_found = False
        total_prune_time = 0.
        # mark previous fit's computed feature importances here.
        if prev_importance_df is not None:
            fn_args['prev_importance_df'] = prev_importance_df
            fn_args['using_prev_fit_fi'] = set(prev_importance_df[prev_importance_df['n'] > 0].index.tolist())
        while self.time_limit > round_time_budget + self.model_fit_time:
            candidate_features, importance_df, prune_time = self.compute_next_candidate_round(**fn_args)
            # HACK: Line below is needed to get this working with n-repeated bagged models. Related to feature ordering.
            candidate_features = [feature for feature in fn_args['X'].columns.tolist() if feature in candidate_features]
            total_prune_time = total_prune_time + prune_time
            candidate_set, best_set, prev_candidate_set = set(candidate_features), set(fn_args['features']), set(prev_candidate_features)
            candidate_found = candidate_set != best_set and candidate_set != prev_candidate_set and len(candidate_set) > 0
            all_features_evaluated = len(importance_df[importance_df['importance'].isna()]) == 0
            fn_args['prev_importance_df'] = importance_df
            if candidate_found or all_features_evaluated:
                break
        logger.log(30, f"\tCandidate generation time: ({round(total_prune_time, 2)}s), Cardinality: {len(candidate_features)}")
        return candidate_features, importance_df, candidate_found, total_prune_time

    def compute_next_candidate_round(self, X: pd.DataFrame, y: pd.Series, model: AbstractModel, time_budget: float, features: List[str],
                                     n_subsample: int, min_fi_samples: int, max_fi_samples: int, prune_ratio: float, prune_threshold: float,
                                     prev_importance_df: pd.DataFrame = None, prioritized: Set[str] = set(), using_prev_fit_fi: Set[str] = set(),
                                     weighted: bool = True) -> Tuple[List[str], pd.DataFrame, float]:
        """
        Compute permutation feature importance for as many features as possible under time_budget. Ensure each returned feature importance
        scores are computed from at least n_sample datapoints. Update self.time_limit to account for feature importance computation time.
        """
        # determine how many subsamples and shuffles to use for feature importance calculation
        time_start = time.time()
        n_features = len(features)
        n_subsample = min(n_subsample, len(X))
        n_sample = max(min_fi_samples, min(max_fi_samples, len(X)))
        n_shuffle = min(np.ceil(n_sample / n_subsample).astype(int), self.max_n_shuffle)
        expected_single_feature_time = self.compute_expected_fi_time_single(X, self.model_predict_time, n_subsample, n_sample)
        auto_threshold = len(prioritized) > 0
        features = sort_features_by_priority(features, prioritized, prev_importance_df, using_prev_fit_fi)

        # if we do not have enough time to evaluate feature importance for all features, do so only for some (first n_evaluated_features elements of features)
        n_evaluated_features = max([i for i in range(0, n_features+1) if i * expected_single_feature_time <= time_budget])
        if n_evaluated_features == 0:
            prune_time = time.time() - time_start
            self.time_limit = self.time_limit - prune_time
            self._fi_time_elapsed = self._fi_time_elapsed + prune_time
            return features, unevaluated_fi_df_template(features), prune_time
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
        importance_df = merge_importance_dfs(prev_importance_df, importance_df, using_prev_fit_fi)

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

    def compute_next_candidate_given_fi(self, importance_df: pd.DataFrame, prune_threshold: float, prune_ratio: float, time_budget: float,
                                        weighted: bool) -> List[str]:
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

    def compute_expected_fi_time_single(self, X_fi: pd.DataFrame, model_predict_time: float, n_subsample: int, n_total_sample: int) -> float:
        """
        Return the expected time to compute permutation feature importance for a single feature on n_total_sample datapoints.
        """
        n_subsample = min(n_subsample, len(X_fi))
        n_shuffle = min(np.ceil(n_total_sample / n_subsample).astype(int), self.max_n_shuffle)
        return self.safety_time_multiplier * model_predict_time * (n_subsample / len(X_fi)) * n_shuffle

    def compute_time_budget_fi(self, X_fi: pd.DataFrame, n_subsample: int, **kwargs):
        """
        Return the time that a single feature importance computation round can take up to. Currently the minimum of 5 minutes
        and the time it takes to fully evaluated minimum of 50 features or the number of features in the dataset.
        """
        min_fi_samples = kwargs.get('min_fi_samples', 10000)
        max_fi_samples = kwargs.get('max_fi_samples', 100000)
        n_total_samples = max(min_fi_samples, min(max_fi_samples, len(X_fi)))
        fi_time_single = self.compute_expected_fi_time_single(X_fi, self.model_predict_time, n_subsample, n_total_samples)
        return min(fi_time_single * min(len(X_fi.columns), 50), self.max_time_budget_fi)

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
            kwargs['save_bag_folds'] = True
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

    def get_extra_fn_args(self, **kwargs) -> dict:
        return {
            'weighted': kwargs.get('weighted', True),
            'min_fi_samples': kwargs.get('min_fi_samples', 10000),
            'max_fi_samples': kwargs.get('max_fi_samples', 100000)
        }

    def consider_subsampling(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.DataFrame, n_train_subsample: int,
                             **kwargs) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, bool, dict]:
        """
        If using a proxy model and dataset size is larger than n_train_subsample, subsample data to make model training faster.
        If the proxy model is bagged and we have a lot of data, use a non-bagged version instead since it is ~10x faster to train.
        Update fit and predict time estimates accordingly.
        """
        X_train, y_train = X, y
        if len(X) > n_train_subsample:
            subsampled = True
            X_train = X.sample(n_train_subsample, random_state=self.rng.integers(low=0, high=1e5))
            y_train = y.loc[X_train.index]
            if kwargs.pop('replace_bag', True) and self.is_bagged and len(X) >= n_train_subsample * 1.2:
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

    def setup(self, X: pd.DataFrame, y: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, n_train_subsample: int, prune_threshold: float,
              kwargs: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool, bool, dict]:
        """
        Modify training data, validation data, and model fit kwargs appropriately by subsampling, adding noise columns, replacing bagged
        models, and more.
        """
        X, y, X_val, y_val, subsampled, kwargs = self.consider_subsampling(X, y, X_val, y_val, n_train_subsample, **kwargs)
        auto_threshold = prune_threshold is None
        if auto_threshold:
            kwargs['feature_metadata'] = deepcopy(kwargs['feature_metadata']) if 'feature_metadata' in kwargs else None
            X = add_noise_column(X, self.noise_prefix, self.rng, feature_metadata=kwargs.get('feature_metadata', None))
            X_val = add_noise_column(X_val, self.noise_prefix, self.rng, feature_metadata=kwargs.get('feature_metadata', None))
        X_fi, y_fi = (X, y) if self.is_bagged else (X_val, y_val)
        if self.is_bagged and self.original_model._child_oof and self.model_fit_time is not None:
            self.model_fit_time = self.model_fit_time * kwargs['k_fold']
        return X, y, X_val, y_val, X_fi, y_fi, auto_threshold, subsampled, kwargs
