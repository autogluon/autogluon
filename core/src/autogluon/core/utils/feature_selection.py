from copy import deepcopy
import logging
import time
import traceback
from typing import List, Set, Tuple, Union
import uuid

import numpy as np
import pandas as pd

from autogluon.common.features.types import R_FLOAT

from ..models.abstract.abstract_model import AbstractModel
from ..models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from ..utils.exceptions import TimeLimitExceeded
from ..utils.utils import generate_train_test_split, unevaluated_fi_df_template

logger = logging.getLogger(__name__)


def add_noise_column(X: pd.DataFrame, rng: np.random.Generator, noise_columns: List[str] = None, count: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create a copy of dataset X with extra synthetic columns generated from standard normal distribution.
    """
    X = X.copy()
    if noise_columns is None:
        noise_columns = [str(uuid.uuid4()) for _ in range(1, count+1)]
    for col_name in noise_columns:
        noise = rng.standard_normal(len(X))
        X[col_name] = noise
    return X, noise_columns


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
    evaluated_new_only_rows = evaluated_new_rows.loc[set(evaluated_new_only + evaluated_new_first_time)]
    # for features with info on both df_new and whose df_old feature importance came from the current model, return combined statistics
    evaluated_both_rows = pd.DataFrame()
    evaluated_both_rows_new = evaluated_new_rows.loc[evaluated_both].sort_index()
    evaluated_both_rows_old = evaluated_old_rows.loc[evaluated_both].sort_index()
    mean_old, mean_new = evaluated_both_rows_old['importance'], evaluated_both_rows_new['importance']
    stddev_old, stddev_new = evaluated_both_rows_old['stddev'], evaluated_both_rows_new['stddev']
    n_old, n_new = evaluated_both_rows_old['n'], evaluated_both_rows_new['n']
    evaluated_both_rows['importance'] = (n_old * mean_old + n_new * mean_new) / (n_old + n_new)
    # Refer to https://math.stackexchange.com/questions/2971315/how-do-i-combine-standard-deviations-of-two-groups
    evaluated_both_rows['stddev'] = (((n_old - 1) * stddev_old ** 2 + (n_new - 1) * stddev_new ** 2) / (n_old + n_new - 1) +
                                     (n_old * n_new * (mean_old - mean_new) ** 2) / ((n_old + n_new) * (n_old + n_new - 1))) ** 0.5
    evaluated_both_rows['p_value'] = None
    evaluated_both_rows['n'] = n_old + n_new
    # remove features evaluated in df_new from using_prev_fit_fi if they exist
    using_prev_fit_fi.difference_update(evaluated_new_rows.index.tolist())
    result = pd.concat([evaluated_both_rows, evaluated_new_only_rows, evaluated_old_only_rows, evaluated_neither_rows]).sort_values('importance')
    assert len(result) == len(df_new), "Length of the updated DataFrame must be equal to the inputted DataFrame."
    return result


def sort_features_by_priority(features: List[str], prev_importance_df: pd.DataFrame, using_prev_fit_fi: Set[str]) -> List[str]:
    """
    Return a list of features sorted by feature importance calculation priority in ascending order.
    If prev_importance_df does not exist and not using noise prune_threshold, return the original list.
    If prev_importance_df exists, features whose importance scores have never been calculated are
    prioritized first followed by features whose importance scores are from previous fitted models and
    lastly the rest of the features sorted by previous importance scores estimates in ascending order. If using
    noise prune_threshold, noise columns are prioritized first since their scores are needed to determine
    the pruning threshold.
    """
    is_first_run = prev_importance_df is None
    if not is_first_run:
        prev_deleted_features = [feature for feature in prev_importance_df.index if feature not in features]
        prev_importance_df = prev_importance_df.drop(prev_deleted_features)
        unevaluated_rows = prev_importance_df[prev_importance_df['importance'].isna()]
        prev_fit_evaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
                                                      (prev_importance_df.index.isin(using_prev_fit_fi))].sort_values(by='importance')
        curr_fit_evaluated_rows = prev_importance_df[~(prev_importance_df['importance'].isna()) &
                                                     ~(prev_importance_df.index.isin(using_prev_fit_fi))].sort_values(by='importance')
        features = unevaluated_rows.index.tolist() + prev_fit_evaluated_rows.index.tolist() + curr_fit_evaluated_rows.index.tolist()
    return features


class FeatureSelector:
    def __init__(self, model: AbstractModel, time_limit: float, problem_type: str, seed: int = 0, raise_exception=False) -> None:
        """
        Parameters
        ----------
        model : AbstractModel
            Model to perform permutation feature importance recursive feature elimination with.
        time_limit : float
            Time budget for the entire feature selection procedure.
        problem_type : str
            Problem type (Ex. binary, regression, ...).
        seed : int, default 0
            Random seed for generating reproducible results.
        raise_exception : bool, default False
            Whether to crash AutoGluon if there is an error in feature selection. If False, return the current best feature subset.
        """
        # TODO: Make this work with unlabelled data
        assert time_limit is not None, "Time limit cannot be unspecified."
        self.is_bagged = isinstance(model, BaggedEnsembleModel)

        self.model_class = model.__class__
        self.model_params = model.get_params()
        self.model_name = "FeatureSelector_" + self.model_params['name']
        self.model_params['name'] = self.model_name
        if self.is_bagged:
            # required for feature importance computation
            self.model_params['hyperparameters']['use_child_oof'] = False
            self.model_params['hyperparameters']['save_bag_folds'] = True
        del model

        self.time_limit = time_limit
        self.time_start = time.time()
        self.problem_type = problem_type
        self.rng = np.random.default_rng(seed)
        self.fit_score_time = None
        self.model_predict_time = None
        self.attempted_removals = set()
        self.replace_bag = False
        self.max_n_shuffle = 20
        self.min_prune_ratio = 0.01
        self.raise_exception = raise_exception

    def select_features(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None, n_train_subsample: int = 50000,
                        n_fi_subsample: int = 10000, prune_threshold: float = 'noise', prune_ratio: float = 0.05, stopping_round: int = 10,
                        min_improvement: float = 1e-6, max_fits: int = None, **kwargs) -> List[str]:
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
        X_val : pd.DataFrame, default None
            Validation data features. Can be None.
        y_val : pd.Series, default None
            Validation data labels. Can be None.
        n_train_subsample : int, default 50000
            If the training dataset has more than this amount of datapoints, sample this many datapoints without replacement and use
            them as feature selection model training data. If None, do not use subsampling.
        n_fi_subsample : int, default 10000
            Sample this many data points and shuffle them when computing permutation feature importance. If this number is higher than
            the number of feature importance data points, set n_fi_subsample = number of feature importance datapoints.
        prune_threshold : Tuple[float,str], default 'noise'
            Consider features whose feature importance scores are below this threshold for pruning. Can be one of ['noise', 'none', float]. If set
            to 'noise', a synthetic columns whose values come from standard normal distribution will be appended to the original dataset, and any
            features whose feature importance score is lower than the synthetic feature's score will be considered for pruning. If set to 'none',
            all features will be considered for pruning (where up to prune_ratio features are pruned at a time). If set to float, any feature whose
            feature importance is lower than that threshold will be considered for pruning.
        prune_ratio : float, default 0.05
            Prune up to prune_ratio * number of current features at once whose feature importance scores are below prune_threshold when
            generating new feature subset candidates. At least one feature is always removed if there are any feature whose importance
            score is below prune_threshold.
        stopping_round : int, default 10
            If the validation scores of models fit on pruned data do not improve for stopping_round feature pruning rounds, end the pruning procedure.
            If None, continue feature pruning until time is up.
        min_improvement : int, default = 1e-6
            The newly fitted model's validation score must be >= (1 + min_improvement) * best validation score seen so far for its input
            feature subset to be considered to be superior to the previous feature subset.
        max_fits : int, default None
            If this many models have been fitted during feature pruning, exit the feature pruning loop. Can potentially prevent overfitting.
            We refit the model using the remaining features after each round of feature pruning. If None, continue feature pruning until time is up.

        Returns
        -------
        candidate_features : List[str]
            Optimal feature subset selected by this method. Set to original features if no features are below pruning threshold or we run
            out of time before finding a better feature subset.
        """
        logger.log(20, f"Performing feature pruning with model: {self.model_name}, total time limit: {round(self.time_limit, 2)}s, "
                       f"stop threshold: {stopping_round}, prune ratio: {prune_ratio}, prune threshold: {prune_threshold}.")
        original_features = X.columns.tolist()
        if len(original_features) <= 1:
            logger.log(20, f"\tSkipping feature pruning since there is less than 2 features in the dataset.")
            return original_features
        X, y, X_val, y_val, X_fi, y_fi, prune_threshold, noise_columns, feature_metadata = self.setup(X=X, y=y, X_val=X_val, y_val=y_val,
                                                                                                      n_train_subsample=n_train_subsample,
                                                                                                      prune_threshold=prune_threshold, **kwargs)
        kwargs['feature_metadata'] = feature_metadata
        try:
            index = 1
            candidate_features = X.columns.tolist()
            best_info = {'features': candidate_features, 'index': 1, 'model': None, 'score': None}
            curr_model, score, fit_score_time = self.fit_score_model(X, y, X_val, y_val, candidate_features, f"{self.model_name}_1", **kwargs)
            best_info['model'], best_info['score'] = curr_model, score

            time_budget_fi = self.compute_time_budget_fi(X_fi=X_fi, n_subsample=n_fi_subsample, **kwargs)
            logger.log(20, f"\tExpected model fit time: {round(fit_score_time, 2)}s, and expected candidate generation time: {round(time_budget_fi, 2)}s.")
            logger.log(20, f"\tRound 1 of feature pruning model fit ({round(fit_score_time, 2)}s):\n"
                           f"\t\tValidation score of the model fit on original features is ({round(best_info['score'], 4)}).")
            time_remaining = self.time_limit - (time.time() - self.time_start)
            if time_remaining < self.fit_score_time + time_budget_fi:
                logger.warning(f"\tNo time to perform the next pruning round (remaining: {time_remaining}, needed: {self.fit_score_time + time_budget_fi}).")
                raise TimeLimitExceeded

            importance_df = None
            while True:
                index = index + 1
                model_name = f"{self.model_name}_{index}"
                prioritize_fi = set(noise_columns)
                fn_args = {'X': X_fi, 'y': y_fi, 'model': best_info['model'], 'time_budget': time_budget_fi, 'features': best_info['features'],
                           'n_subsample': n_fi_subsample, 'prune_threshold': prune_threshold, 'prune_ratio': prune_ratio, 'prioritized': prioritize_fi}
                fn_args.update(self.get_extra_fn_args(**kwargs))
                candidate_features, importance_df, success, prune_time = self.compute_next_candidate(fn_args=fn_args, round_time_budget=time_budget_fi,
                                                                                                     prev_best_features=best_info['features'],
                                                                                                     prev_importance_df=importance_df)
                if not success:
                    logger.log(20, f"\tTime is up while computing feature importance or there are no more features to prune. Ending...")
                    break
                curr_model, score, fit_score_time = self.fit_score_model(X, y, X_val, y_val, candidate_features, model_name, **kwargs)

                new_feature_count = len(candidate_features) - (1 if len(noise_columns) > 0 else 0)
                prev_feature_count = len(best_info['features']) - (1 if len(noise_columns) > 0 else 0)
                if score >= best_info['score'] * (1 + min_improvement):
                    logger.log(20, f"\tRound {index} of feature pruning model fit ({round(fit_score_time, 2)}s):\n"
                                   f"\t\tValidation score of the current model fit on {new_feature_count} features ({round(score, 4)}) is better than "
                                   f"validation score of the best model fit on {prev_feature_count} features ({round(best_info['score'], 4)}). Updating model.")
                    best_info['model'].delete_from_disk(silent=True)
                    best_info = {'model': curr_model, 'features': candidate_features, 'score': score, 'index': index}
                else:
                    logger.log(20, f"\tRound {index} of feature pruning model fit ({round(fit_score_time, 2)}s):\n"
                                   f"\t\tValidation score of the current model fit on {new_feature_count} features ({round(score, 4)}) is not better than "
                                   f"validation score of the best model fit on {prev_feature_count} features ({round(best_info['score'], 4)}). Retrying.")
                    curr_model.delete_from_disk(silent=True)

                time_remaining = self.time_limit - (time.time() - self.time_start)
                if max_fits is not None and index >= max_fits:
                    logger.log(20, f"\tReached maximum number of allowed fits, {max_fits}, during feature selection. Ending...")
                    break
                if stopping_round is not None and index - best_info['index'] >= stopping_round:
                    logger.log(20, f"\tScore has not improved for {stopping_round} feature pruning rounds. Ending...")
                    break
                if time_remaining <= self.fit_score_time + prune_time:
                    logger.log(20, f"\tInsufficient time to finish next pruning round. Ending...")
                    break
        except TimeLimitExceeded:
            logger.log(20, f"\tTime limit exceeded while pruning features. Ending...")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"\tERROR: Exception raised during feature pruning. Reason: {e}. Ending...")
            if self.raise_exception:
                raise e

        if len(noise_columns) > 0:
            best_info['features'] = [feature for feature in best_info['features'] if feature not in noise_columns]
        if isinstance(best_info['model'], AbstractModel):
            best_info['model'].delete_from_disk(silent=True)
        logger.log(20, f"\tSuccessfully ended prune loop after {index} feature pruning rounds ({round(time.time() - self.time_start, 2)}s).")
        logger.log(20, f"\tFeature count before/after feature pruning: {len(original_features)} -> {len(best_info['features'])}.")
        return best_info['features']

    def compute_next_candidate(self, fn_args: dict, round_time_budget: float, prev_best_features: List[str], prev_importance_df: pd.DataFrame
                               ) -> Tuple[List[str], pd.DataFrame, bool, float]:
        """
        While time allows, repeatedly compute feature importance and generate candidate feature subsets using a fixed time budget.
        If at least self.min_prune_ratio of the features can be pruned or all feature importances are calculated, return. If less than
        self.min_prune_ratio of the features can be pruned but not all importance scores have been calculated, repeat the procedure.
        Note: If we have feature importance information from previously fitted model, make use of them since we might not have time this
        iteration to evaluate importance scores for all features. Any feature importance calculated within this call of compute_next_candidate
        will override previous feature importance info.
        """
        candidate_features = prev_best_features
        importance_df = unevaluated_fi_df_template(candidate_features)
        candidate_found = False
        total_prune_time = 0.
        # mark previous fit's computed feature importances here.
        if prev_importance_df is not None:
            fn_args['prev_importance_df'] = prev_importance_df
            fn_args['using_prev_fit_fi'] = set(prev_importance_df[prev_importance_df['n'] > 0].index.tolist())
        while self.time_limit - (time.time() - self.time_start) > round_time_budget + self.fit_score_time:
            candidate_features, importance_df, prune_time = self.compute_next_candidate_round(**fn_args)
            # HACK: Line below is needed to get this working with n-repeated bagged models. Related to feature ordering.
            candidate_features = [feature for feature in fn_args['X'].columns.tolist() if feature in candidate_features]
            candidate_found = len(candidate_features) > 0 and len(candidate_features) <= (1. - self.min_prune_ratio) * len(prev_best_features)
            all_features_evaluated = len(importance_df[importance_df['importance'].isna()]) == 0
            fn_args['prev_importance_df'] = importance_df
            total_prune_time = total_prune_time + prune_time
            if candidate_found or all_features_evaluated:
                break
        logger.log(15, f"\tCandidate generation time: ({round(total_prune_time, 2)}s), Cardinality: {len(candidate_features)}")
        return candidate_features, importance_df, candidate_found, total_prune_time

    def compute_next_candidate_round(self, X: pd.DataFrame, y: pd.Series, model: AbstractModel, time_budget: float, features: List[str],
                                     n_subsample: int, min_fi_samples: int, max_fi_samples: int, prune_ratio: float, prune_threshold: float,
                                     prev_importance_df: pd.DataFrame = None, prioritized: Set[str] = set(), using_prev_fit_fi: Set[str] = set(),
                                     weighted: bool = True) -> Tuple[List[str], pd.DataFrame, float]:
        """
        Compute permutation feature importance for as many features as possible under time_budget. Ensure each returned feature importance
        scores are computed from at least n_sample datapoints.
        """
        # determine how many subsamples and shuffles to use for feature importance calculation
        time_start = time.time()
        n_features = len(features)
        n_subsample = min(n_subsample, len(X))
        n_total_sample = max(min_fi_samples, min(max_fi_samples, len(X)))
        n_shuffle = min(np.ceil(n_total_sample / n_subsample).astype(int), self.max_n_shuffle)
        single_feature_fi_time = self.compute_expected_fi_time_single(X_fi=X, model_predict_time=self.model_predict_time,
                                                                      n_subsample=n_subsample, n_total_sample=n_total_sample)
        noise_threshold = len(prioritized) > 0
        features = sort_features_by_priority(features=features, prev_importance_df=prev_importance_df, using_prev_fit_fi=using_prev_fit_fi)
        if noise_threshold:
            features = list(prioritized) + [feature for feature in features if feature not in prioritized]

        # if we do not have enough time to evaluate feature importance for all features, do so only for some (first n_evaluated_features elements of features)
        n_evaluated_features = max([i for i in range(0, n_features+1) if i * single_feature_fi_time + self.model_predict_time <= time_budget])
        if n_evaluated_features == 0:
            prune_time = time.time() - time_start
            return features, unevaluated_fi_df_template(features), prune_time
        evaluated_features = features[:n_evaluated_features]
        unevaluated_features = features[n_evaluated_features:]
        time_budget_fi = time_budget - (time.time() - time_start)
        logger.log(15, f"\tComputing feature importance for {n_evaluated_features}/{n_features} features with {n_shuffle} shuffles.")
        fi_kwargs = {'X': X, 'y': y, 'num_shuffle_sets': n_shuffle, 'subsample_size': n_subsample, 'features': evaluated_features,
                     'time_limit': time_budget_fi, 'silent': True, 'random_state': self.rng.integers(low=0, high=1e5)}
        fi_kwargs.update({'is_oof': True} if self.is_bagged else {})
        # FIXME: Right now the upper bound on the number of features we evaluate feature importance at once is determined by our expected feature
        # importance computation time. While this estimate is relatively accurate on most datasets, on some high dimensional datasets it underestimates
        # the time needed to evaluate n_shuffles of permutations and ends up only evaluating a few shuffles. Consider making a feature importance
        # method that parallelizes across individual shuffles instead.
        evaluated_df = model.compute_feature_importance(**fi_kwargs)
        if self.is_bagged:
            # If the bagged model includes 5 models and we evaluate a single permutation feature importance shuffle, the above method returns n=5 instead of 1.
            evaluated_df['n'] = (evaluated_df['n'] // len(model.models)).clip(lower=1)

        # if we could not compute feature importance for all features and previous feature importance estimates exist, use them
        importance_df = pd.concat([evaluated_df, unevaluated_fi_df_template(unevaluated_features)])
        importance_df = merge_importance_dfs(prev_importance_df, importance_df, using_prev_fit_fi)

        # if using noise threshold, threshold is the mean of noise column importance score
        if noise_threshold:
            noise_rows = importance_df[importance_df.index.isin(prioritized)]
            importance_df = importance_df.drop(prioritized)
            prune_threshold = noise_rows['importance'].mean()

        # use importance_df to generate next candidate features
        candidate_features = self.compute_next_candidate_given_fi(importance_df=importance_df, prune_threshold=prune_threshold,
                                                                  prune_ratio=prune_ratio, weighted=weighted)

        # if noise columns exist, they should never be removed
        if noise_threshold:
            candidate_features = candidate_features + list(prioritized)
            importance_df = pd.concat([importance_df, noise_rows])

        feature_selection_time = time.time() - time_start
        return candidate_features, importance_df.sort_values(by='importance', axis=0), feature_selection_time

    def compute_next_candidate_given_fi(self, importance_df: pd.DataFrame, prune_threshold: float, prune_ratio: float, weighted: bool) -> List[str]:
        """
        Keep features whose importance scores are above threshold or have not yet had a chance to be calculated,
        as well as some features whose importance scores are below threshold if more than prune_ratio * num features
        features are below threshold. In the latter case, randomly sample without replacement from features whose
        importance scores are below threshold until removal candidate configuration that has not yet been tried
        is encountered. Give higher probability to features whose scores are lower than others when sampling.
        """
        n_remove = max(1, int(prune_ratio * len(importance_df)))
        above_threshold_rows = importance_df[(importance_df['importance'] > prune_threshold) | (importance_df['importance'].isna())]
        below_threshold_rows = importance_df[importance_df['importance'] <= prune_threshold].sort_values(by='importance', axis=0, ascending=True)
        logger.log(15, f"\tNumber of identified features below prune threshold {round(prune_threshold, 4)}: {len(below_threshold_rows)}/{len(importance_df)}")
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
        for _ in range(50):
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
        Assumes baseline validation score has already been computed. TODO: Take into account speedup from parallel feature
        processing and slowdown from permutation.
        """
        n_subsample = min(n_subsample, len(X_fi))
        n_shuffle = min(np.ceil(n_total_sample / n_subsample).astype(int), self.max_n_shuffle)
        return model_predict_time * ((n_subsample / len(X_fi)) * n_shuffle)

    def compute_time_budget_fi(self, X_fi: pd.DataFrame, n_subsample: int, **kwargs):
        """
        Return the time that a single feature importance computation round can take up to. Currently the time it
        takes to fully evaluated minimum of 50 features or the number of features in the dataset.
        """
        min_fi_samples = kwargs.get('min_fi_samples', 10000)
        max_fi_samples = kwargs.get('max_fi_samples', 50000)
        n_total_samples = max(min_fi_samples, min(max_fi_samples, len(X_fi)))
        baseline_time = self.model_predict_time
        fi_time_single = self.compute_expected_fi_time_single(X_fi=X_fi, model_predict_time=self.model_predict_time,
                                                              n_subsample=n_subsample, n_total_sample=n_total_samples)
        return fi_time_single * min(len(X_fi.columns), 50) + baseline_time

    def fit_score_model(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                        features: List[str], model_name: str, **kwargs) -> Tuple[AbstractModel, float, float]:
        """
        Fits and scores a model over the given feature subset (ex. features remaining in a particular feature pruning round).
        Returns the fitted model, its score, and time elapsed. If this is the first time we are fitting a model in the pruning
        procedure, save time and score statistics.
        """
        time_start = time.time()
        model = self.model_class(**self.model_params)
        if self.replace_bag:
            model = model.convert_to_template_child()
        model.rename(model_name)
        X = X[features]
        X_val = None if self.is_bagged else X_val[features]
        if 'time_limit' in kwargs:
            time_remaining = self.time_limit - (time.time() - self.time_start)
            kwargs['time_limit'] = time_remaining
        model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)
        time_fit = time.time() - time_start
        score = model.score_with_oof(y) if self.is_bagged else model.score(X=X_val, y=y_val)
        time_fit_score = time.time() - time_start
        if self.fit_score_time is None:
            self.fit_score_time = time_fit_score
        if self.model_predict_time is None:
            self.model_predict_time = model.predict_time if self.is_bagged else time_fit_score - time_fit
        return model, score, time_fit_score

    def get_extra_fn_args(self, **kwargs) -> dict:
        return {
            'weighted': kwargs.get('weighted', True),
            'min_fi_samples': kwargs.get('min_fi_samples', 10000),
            'max_fi_samples': kwargs.get('max_fi_samples', 50000),
        }

    def setup(self, X: pd.DataFrame, y: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.DataFrame, n_train_subsample: int, prune_threshold: float,
              **kwargs: dict) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Union[float, str], List[str]]:
        """
        Modify training data, validation data, and model fit kwargs appropriately by subsampling, adding noise columns, replacing bagged
        models, and more.
        """
        # subsample training data
        min_fi_samples = kwargs.get('min_fi_samples', 10000)
        random_state = self.rng.integers(low=0, high=1e5)
        if n_train_subsample is not None and len(X) > n_train_subsample:
            logger.log(20, f"\tNumber of training samples {len(X)} is greater than {n_train_subsample}. Using {n_train_subsample} samples as training data.")
            drop_ratio = 1. - n_train_subsample / len(X)
            X_train, _, y_train, _ = generate_train_test_split(X=X, y=y, problem_type=self.problem_type, random_state=random_state, test_size=drop_ratio)
        else:
            X_train, y_train = X, y

        # replace bagged model with its child model for the proxy model if replace_bag=True (overrides subsampling if triggered)
        if n_train_subsample is None:
            trigger_replace_bag = kwargs.get('replace_bag', False) and self.is_bagged
        else:
            trigger_replace_bag = kwargs.get('replace_bag', True) and self.is_bagged and len(X) > n_train_subsample + min_fi_samples
        if trigger_replace_bag:
            logger.log(20, f"\tFeature selection model is bagged and replace_bag=True. Using a non-bagged version of the model for feature selection.")
            val_ratio = 1. - n_train_subsample / len(X) if n_train_subsample is not None else 0.25
            X_train, X_val, y_train, y_val = generate_train_test_split(X=X, y=y, problem_type=self.problem_type, random_state=random_state, test_size=val_ratio)
            self.is_bagged = False
            self.replace_bag = True

        # Be more lenient with feature importance computation shuffles for very high dimensional datasets for time's sake
        if len(X_train.columns) > 1000:
            self.max_n_shuffle = self.max_n_shuffle // 2

        # set prune_threshold and optionally modify feature_metadata
        noise_columns = []
        feature_metadata = deepcopy(kwargs.get('feature_metadata', None))
        if prune_threshold == 'none':
            prune_threshold = float('inf')
        elif prune_threshold == 'noise':
            X_train, noise_columns = add_noise_column(X=X_train, rng=self.rng)
            if feature_metadata is not None:
                for noise_column in noise_columns:
                    feature_metadata.type_map_raw[noise_column] = R_FLOAT
            if isinstance(X_val, pd.DataFrame):
                X_val, _ = add_noise_column(X=X_val, rng=self.rng, noise_columns=noise_columns)
        else:
            assert isinstance(prune_threshold, float), "prune_threshold must be float, 'noise', or 'none'."
        X_fi, y_fi = (X_train, y_train) if self.is_bagged else (X_val, y_val)
        return X_train, y_train, X_val, y_val, X_fi, y_fi, prune_threshold, noise_columns, feature_metadata
