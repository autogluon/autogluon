from typing import Sequence, Tuple
from autogluon.core.features.feature_metadata import FeatureMetadata
from autogluon.core.models.abstract.abstract_model import AbstractModel
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.core.models.ensemble.stacker_ensemble_model import StackerEnsembleModel
from copy import deepcopy
import logging
from autogluon.core.utils.utils import unevaluated_fi_df_template
from autogluon.core.utils.exceptions import TimeLimitExceeded
from autogluon.core.features.types import R_FLOAT
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)


def add_noise_column(X: pd.DataFrame, prefix: str, count: int = 1, feature_metadata: FeatureMetadata = None, seed: int = 0) -> pd.DataFrame:
    """
    Create a copy of dataset X with extra synthetic columns generated from standard normal distribution.
    """
    if X is None:
        return None
    X = X.copy()
    rng = np.random.default_rng(seed=seed)
    for i in range(1, count+1):
        col_name = f"{prefix}_{i}"
        if feature_metadata is not None:
            feature_metadata.type_map_raw[col_name] = R_FLOAT
        noise = rng.standard_normal(len(X))
        X[col_name] = noise
    return X


def compute_prune_threshold_from_noise(importance_df: pd.DataFrame, prioritized: Sequence[str]) -> float:
    noise_rows = importance_df[importance_df.index.isin(prioritized)]
    importance_df = importance_df.drop(prioritized)
    return noise_rows['importance'].mean()


class FeatureSelector:
    def __init__(self, model: AbstractModel, time_limit: float, keep_models: bool = False) -> None:
        self.original_model = model
        self.base_model = model.convert_to_template()
        self.base_model.rename(f'FeatureSelector_{self.base_model.name}')
        if time_limit is None:
            raise AssertionError("Time limit cannot be unspecified.")
        self.original_time_limit = time_limit
        self.time_limit = time_limit
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

    def select_features(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, train_subsample_size: int = 50000,
                        fi_subsample_size: int = 5000, prune_ratio: float = 0.05, prune_threshold: float = None, stop_threshold: int = 1,
                        min_fi_samples: int = 10000, max_fits: int = 5, **kwargs) -> Tuple[Sequence[str], Sequence[pd.DataFrame]]:
        # TODO: 1. Try not exiting the feature selection immediately when score doesn't improve once
        # TODO: 2. Add ability to attach extra metadata on automlbenchmark (check what features are being pruned)
        # TODO: 3. Add logic where if we have a LOT of data, we don't do bagging and use leftover data for validation dataset
        # TODO: 4. Consider repeated bagging for initial model fits (use all data first) added (n-repeat for model refit score calculation)
        # subsample training data and optionally add noise features to dataset
        original_features = X.columns.tolist()
        auto_threshold = prune_threshold is None
        refit_at_end = auto_threshold and not self.is_proxy_model
        noise_prefix = 'AG_normal_noise'
        if self.is_proxy_model:
            X = X.sample(train_subsample_size, random_state=0) if train_subsample_size < len(X) else X
            y = y.loc[X.index]
        if auto_threshold:
            if kwargs.get('feature_metadata', None) is not None:
                kwargs['feature_metadata'] = deepcopy(kwargs['feature_metadata'])
            X = add_noise_column(X, prefix=noise_prefix, feature_metadata=kwargs.get('feature_metadata', None))
            X_val = add_noise_column(X_val, prefix=noise_prefix, feature_metadata=kwargs.get('feature_metadata', None))

        index = 1
        candidate_features = X.columns.tolist()
        X_fi, y_fi = (X, y) if self.is_bagged else (X_val, y_val)
        best_info = {'features': candidate_features, 'index': index, 'model': self.original_model, 'score': self.original_val_score}
        logger.log(30, f"\tPerforming V1 model feature selection with model: {self.original_model.name}, total time limit: {round(self.time_limit, 2)}s, " +
                       f"max fits: {max_fits}, stop threshold: {stop_threshold}, prune ratio: {prune_ratio}, prune threshold: {prune_threshold}.")
        try:
            minimum_model_fits = 3 if refit_at_end else 2
            if self.is_proxy_model and self.time_limit <= self.model_fit_time * minimum_model_fits:
                logger.log(30, f"\tInsufficient time to perform even a single pruning round.")
                raise TimeLimitExceeded

            # fit proxy model once on the subsampled dataset to serve as scoring reference
            # use original fitted model to compute the first round of feature importance scores, not fitted proxy model
            model_name = f"{self.base_model.name}_{index}"
            model, score, fit_time = self.fit_score_model(deepcopy(self.base_model), X, y, X_val, y_val, candidate_features, model_name, **kwargs)
            if auto_threshold or not self.is_proxy_model:
                best_info['model'], best_info['score'] = model, score

            time_budget_fi = max(0.1 * self.model_fit_time, 10 * self.model_predict_time * min(50, len(X.columns)), 60)
            logger.log(30, f"\tExpected model fit time: {round(self.model_fit_time, 2)}s, and expected candidate generation time: {round(time_budget_fi, 2)}s.")
            logger.log(30, f"\tFit {index} ({fit_time}s): Current score is {score}.")
            if self.is_proxy_model and self.time_limit <= self.model_fit_time * (minimum_model_fits - 1) + time_budget_fi:
                logger.log(30, f"\tInsufficient time to perform even a single pruning round.")
                raise TimeLimitExceeded

            stop_prune = False
            importance_df = None
            while not stop_prune:
                index = index + 1
                model_name = f"{self.base_model.name}_{index}"
                old_candidate_features = candidate_features
                time_start = time.time()
                prioritize_fi = [feature for feature in best_info['features'] if noise_prefix in feature]
                fn_args = {'X': X_fi, 'y': y_fi, 'model': best_info['model'], 'time_budget': time_budget_fi, 'features': best_info['features'],
                           'n_sample': max(min_fi_samples, len(X_fi)), 'n_subsample': fi_subsample_size, 'prev_importance_df': importance_df,
                           'prune_threshold': prune_threshold, 'prune_ratio': prune_ratio, 'prioritized': prioritize_fi}
                candidate_features, importance_df = self.compute_next_candidate(**fn_args)
                # HACK: To get this working with repeated bagged CatBoost and MXNet model, features must have original ordering (...)
                # This is because, for example, CatBoost filters categorical features by indexes and complains if test set features
                # are out of order. This causes an issue with bagged models because X[features] adheres to ordering of features, and
                # parent.features and child.features can have different ordering. However, bagged model organizes features according to
                # parent's ordering. TODO: TALK MORE ABOUT THIS ISSUE
                candidate_features = [feature for feature in X.columns.tolist() if feature in candidate_features]
                self.importance_dfs.append(importance_df)
                feature_selection_time = time.time() - time_start
                self.time_limit = self.time_limit - feature_selection_time
                logger.log(30, f"\tCandidate generation time: ({round(feature_selection_time, 4)}s), Cardinality: {len(candidate_features)}")
                if set(candidate_features) == set(best_info['features']) or set(candidate_features) == set(old_candidate_features) \
                   or len(candidate_features) == 0:
                    logger.log(30, f"\tThere are no more features to prune. Ending...")
                    break
                curr_model, score, fit_time = self.fit_score_model(deepcopy(self.base_model), X, y, X_val, y_val, candidate_features, model_name, **kwargs)

                if score > best_info['score']:
                    message = f"\tFit {index} ({fit_time}s): Current score {score} is better than best score {best_info['score']}. Updating model.\n" +\
                              f"\tOld # Features: {len(best_info['features'])} / New # Features: {len(candidate_features)}."
                    best_info = {'model': curr_model, 'features': candidate_features, 'score': score, 'index': index}
                elif index - best_info['index'] >= stop_threshold:
                    stop_prune = True
                    message = f"\tFit {index} ({fit_time}s): Score has not improved for {stop_threshold} iterations. Current score is {score}. Ending..."
                elif index == max_fits:
                    stop_prune = True
                    message = f"\tFit {index} ({fit_time}s): Maximum number of iterations reached. Current score is {score}. Ending..."
                elif (refit_at_end and self.time_limit <= 2 * self.model_fit_time + feature_selection_time) or \
                     (not refit_at_end and self.time_limit <= self.model_fit_time + feature_selection_time):
                    stop_prune = True
                    message = f"\tFit {index} ({fit_time}s): Insufficient time to finish next pruning round.. Current score is {score}. Ending..."
                else:
                    message = f"\tFit {index} ({fit_time}s): Current score {score} is not better than best score {best_info['score']}. " +\
                            "Retrying fit with a different set of features."
                logger.log(30, message)
        except TimeLimitExceeded:
            logger.log(30, f"\tTime limit exceeded while pruning features. Ending...")
        except Exception as e:
            logger.log(30, f"\tERROR: Exception raised during feature pruning. Reason: {e}. Ending...")

        if auto_threshold:
            best_info['features'] = [feature for feature in best_info['features'] if noise_prefix not in feature]
        if refit_at_end:
            model_name = f"{self.base_model.name}_Final" if self.is_proxy_model else self.base_model.name
            model, score, _ = self.fit_score_model(deepcopy(self.base_model), X, y, X_val, y_val, best_info['features'], model_name, **kwargs)
            best_info['model'], best_info['score'] = model, score

        logger.log(30, f"\tSuccessfully ended prune loop after {index} iterations. Best score: {best_info['score']}.")
        logger.log(30, f"\tFeature Count: {len(original_features)} -> {len(best_info['features'])} ({round(self.original_time_limit - self.time_limit, 2)}s)")
        return best_info['features'], best_info['model']

    def compute_next_candidate(self, X: pd.DataFrame, y: pd.Series, model: AbstractModel, time_budget: float, features: Sequence[str],
                               n_sample=10000, n_subsample=5000, prev_importance_df: pd.DataFrame = None, prune_threshold: float = None,
                               prune_ratio: float = 0.05, prioritized: Sequence[str] = []) -> Tuple[Sequence[str], pd.DataFrame]:
        # determine how many subsamples and shuffles to use for feature importance calculation
        n_features = len(features)
        n_subsample = min(n_subsample, len(X))
        n_shuffle = np.ceil(n_sample / n_subsample).astype(int)
        auto_threshold = len(prioritized) > 0
        is_first_run = prev_importance_df is None

        # features are sorted by feature importance computation priority in ascending order
        if not is_first_run:
            prev_deleted_features = [feature for feature in prev_importance_df.index if feature not in features]
            sorted_features = prev_importance_df.drop(prev_deleted_features).sort_values(by='importance', axis=0, ascending=False).index.tolist()[::-1]
            features = sorted_features + [feature for feature in features if feature not in sorted_features]
        if auto_threshold:
            non_prioritized = [feature for feature in features if feature not in prioritized]
            features = prioritized + non_prioritized

        # if we do not have enough time to evaluate feature importance for all features, do so only for some (first K elements of features)
        expected_single_feature_time = 1.1 * self.model_predict_time * (n_subsample / len(X)) * n_shuffle
        n_evaluated_features = max([i for i in range(1, n_features+1) if i * expected_single_feature_time < time_budget])
        if n_evaluated_features == 0:
            return features, unevaluated_fi_df_template(features)
        evaluated_features = features[:n_evaluated_features]
        unevaluated_features = features[n_evaluated_features:]
        logger.log(30, f"\tComputing feature importance for {n_evaluated_features}/{n_features} features with {n_shuffle} shuffles.")
        evaluated_df = model.compute_feature_importance(X=X, y=y, num_shuffle_sets=n_shuffle, subsample_size=n_subsample, features=evaluated_features,
                                                        is_oof=self.is_bagged, silent=True, time_limit=time_budget)

        # if we could not compute feature importance for all features and previous feature importance estimates exist, use them
        if is_first_run:
            unevaluated_df = unevaluated_fi_df_template(unevaluated_features)
        else:
            unevaluated_df = prev_importance_df[prev_importance_df.index.isin(unevaluated_features)]
        importance_df = pd.concat([evaluated_df, unevaluated_df])

        # if auto_threshold, threshold is the mean of noise column importance score
        if auto_threshold:
            noise_rows = importance_df[importance_df.index.isin(prioritized)]
            importance_df = importance_df.drop(prioritized)
            prune_threshold = noise_rows['importance'].mean()

        # keep features whose importance scores are above threshold or have not had a chance to be calculated
        # only prune up to prune_ratio * n_features features at once
        candidate_features = importance_df[(importance_df['importance'] > prune_threshold) | (importance_df['importance'].isna())].index.tolist()
        removed_features = importance_df[importance_df['importance'] <= prune_threshold].index.tolist()
        n_candidate, n_total = len(candidate_features), len(candidate_features)+len(removed_features)
        logger.log(30, f"\tNumber of features above the pruning threshold {round(prune_threshold, 4)}: {n_candidate}/{n_total}")
        candidate_features = candidate_features + removed_features[:len(removed_features) - max(1, int(prune_ratio * n_features))]

        # if noise columns exist, they should never be removed
        if auto_threshold:
            candidate_features = candidate_features + prioritized
            importance_df = pd.concat([importance_df, noise_rows])
        return candidate_features, importance_df.sort_values(by='importance', axis=0)

    def fit_score_model(self, model: AbstractModel, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                        features: Sequence[str], model_name: str, **kwargs) -> Tuple[AbstractModel, float, float]:
        """
        Fits and scores a model. Updates self.time_limit. Returns the fitted model, its score, and time elapsed.
        If this is the first time we are fitting a model in the pruning procedure, save time and score statistics.
        """
        time_start = time.time()
        X = X[features]
        X_val = None if self.is_bagged else X_val[features]
        model.rename(model_name)
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
        return model, round(score, 4), round(fit_time + predict_time, 4)
