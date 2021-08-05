from typing import Sequence, Tuple
from autogluon.core.models.abstract.abstract_model import AbstractModel
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from copy import deepcopy
import logging
from autogluon.core.utils.exceptions import TimeLimitExceeded
import numpy as np
import pandas as pd
import time

logger = logging.getLogger(__name__)


class ProxyFeatureSelector:
    def __init__(self, model: AbstractModel, time_limit: float) -> None:
        self.fit_model = model
        self.base_model = model.convert_to_template()
        self.base_model.rename(f'Proxy_{self.base_model.name}')
        self.original_time_limit = time_limit
        self.time_limit = time_limit
        if not model.is_valid():
            raise AssertionError("A model must have been fit before feature pruning occurs.")
        self.model_fit_time = model.fit_time
        self.model_predict_time = model.predict_time
        # TODO: can we decide how many subsamples to take based on model.predict_time?
        self.is_bagged = isinstance(model, BaggedEnsembleModel)
        self.importance_dfs = []

    def select_features(self, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, train_subsample_size: int = 50000,
                        fi_subsample_size: int = 5000, prune_ratio: float = 0.05, prune_threshold: float = 0., stop_threshold: int = 1,
                        min_fi_samples: int = 10000, max_fits: int = 5, **kwargs) -> Tuple[Sequence[str], Sequence[pd.DataFrame]]:
        # subsample training data
        X = X.sample(train_subsample_size, random_state=0) if train_subsample_size < len(X) else X
        y = y.loc[X.index]
        X_fi, y_fi = (X, y) if self.is_bagged else (X_val, y_val)
        curr_model = self.fit_model
        time_budget = max(0.1 * self.model_fit_time, self.model_predict_time * min(50, len(X.columns)), 60)
        candidate_features = X.columns.tolist()
        importance_df = None
        index = 1
        best_info = {'features': candidate_features, 'index': index, 'score': None}
        logger.log(30, f"\tPerforming proxy model feature selection with model: {curr_model.name}, total time limit: {round(self.time_limit, 2)}s, " +
                       f"expected model fit time: {round(self.model_fit_time, 2)}s, and expected candidate generation time: {round(time_budget, 2)}s. " +
                       f"max fits: {max_fits}, stop threshold: {stop_threshold}, prune ratio: {prune_ratio}, prune threshold: {prune_threshold}.")
        try:
            if self.time_limit <= self.model_fit_time * 2 + time_budget:
                logger.log(30, f"\tInsufficient time to perform even a single pruning round. Ending...")
                raise TimeLimitExceeded
            # fit proxy model once on the subsampled dataset to serve as scoring reference
            # use original fitted model to compute the first round of feature importance scores, not fitted proxy model
            model, score, fit_time = self.fit_score_model(model=deepcopy(self.base_model), X=X, y=y, X_val=X_val,
                                                          y_val=y_val, features=candidate_features, **kwargs)
            best_info['model'], best_info['score'] = model, score
            logger.log(30, f"\tFit {index} ({fit_time}s): Current score is {score}.")
            stop_prune = False
            while not stop_prune:
                index = index + 1
                old_candidate_features = candidate_features
                time_start = time.time()
                fn_args = {'X': X_fi, 'y': y_fi, 'model': best_info['model'], 'time_budget': time_budget, 'features': best_info['features'],
                           'n_sample': max(min_fi_samples, len(X_fi)), 'n_subsample': fi_subsample_size, 'prev_importance_df': importance_df,
                           'prune_threshold': prune_threshold, 'prune_ratio': prune_ratio}
                candidate_features, importance_df = self.compute_next_candidate(**fn_args)
                self.importance_dfs.append(importance_df)
                feature_selection_time = time.time() - time_start
                self.time_limit = self.time_limit - feature_selection_time
                logger.log(30, f"\tCandidate generation time: ({round(feature_selection_time, 4)}s), Cardinality: {len(candidate_features)}")
                if set(candidate_features) == set(best_info['features']) or set(candidate_features) == set(old_candidate_features):
                    logger.log(30, f"\tThere are no more features to prune. Ending...")
                    break
                curr_model, score, fit_time = self.fit_score_model(model=deepcopy(self.base_model), X=X, y=y, X_val=X_val,
                                                                   y_val=y_val, features=candidate_features, **kwargs)
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
                elif self.time_limit <= self.model_fit_time + feature_selection_time:
                    stop_prune = True
                    message = f"\tFit {index} ({fit_time}s): Insufficient time to finish next pruning round.. Current score is {score}. Ending..."
                else:
                    message = f"\tFit {index} ({fit_time}s): Current score {score} is not better than best score {best_info['score']}. " +\
                            "Retrying fit with a different set of features."
                logger.log(30, message)
        except TimeLimitExceeded:
            logger.log(30, f"\tTime limit exceeded while pruning features. Ending...")
        except Exception as e:
            logger.log(30, f"\tERROR: Exception raised during fit_with_prune. Reason: {e}. Ending...")
        logger.log(30, f"\tSuccessfully ended prune loop after {index} iterations. Best score: {best_info['score']}.")
        logger.log(30, f"\tFeature Count: {len(X.columns.tolist())} -> {len(best_info['features'])}")
        logger.log(30, f"\tPruning Runtime: {round(self.original_time_limit - self.time_limit, 4)}")
        return best_info['features'], self.importance_dfs

    def compute_next_candidate(self, X: pd.DataFrame, y: pd.Series, model: AbstractModel, time_budget: float, features: Sequence[str],
                               n_sample=10000, n_subsample=5000, prev_importance_df: pd.DataFrame = None, prune_threshold: float = 0.,
                               prune_ratio: float = 0.5) -> Tuple[Sequence[str], pd.DataFrame]:
        n_features = len(features)
        n_subsample = min(n_subsample, len(X))
        n_shuffle = np.ceil(n_sample / n_subsample).astype(int)
        if prev_importance_df is not None:
            sorted_features = prev_importance_df.sort_values(by='importance', axis=0, ascending=False).index.tolist()[::-1]
            features = [feature for feature in sorted_features if feature in features] + [feature for feature in features if feature not in sorted_features]
        # if we do not have enough time to evaluate feature importance for all features, do so only for some
        expected_single_feature_time = 1.1 * self.model_predict_time * (n_subsample / len(X)) * n_shuffle
        n_evaluated_features = max([i for i in range(1, n_features+1) if i * expected_single_feature_time < time_budget])
        if n_evaluated_features == 0:
            importance_df = pd.DataFrame({'name': features})
            importance_df['importance'] = None
            importance_df['stddev'] = None
            importance_df['p_value'] = None
            importance_df['n'] = 0
            importance_df.set_index('name', inplace=True)
            importance_df.index.name = None
            return features, importance_df
        evaluated_features = features[:n_evaluated_features]
        unevaluated_features = features[n_evaluated_features:]
        logger.log(30, f"\tComputing feature importance for {n_evaluated_features}/{n_features} features with {n_shuffle} shuffles.")
        evaluated_df = model.compute_feature_importance(X=X, y=y, num_shuffle_sets=n_shuffle, subsample_size=n_subsample,
                                                        features=evaluated_features, is_oof=self.is_bagged, silent=True,
                                                        time_limit=time_budget)
        if prev_importance_df is not None:
            unevaluated_df = prev_importance_df[prev_importance_df.index.isin(unevaluated_features)]
        else:
            unevaluated_df = pd.DataFrame({'name': unevaluated_features})
            unevaluated_df['importance'] = None
            unevaluated_df['stddev'] = None
            unevaluated_df['p_value'] = None
            unevaluated_df['n'] = 0
            unevaluated_df.set_index('name', inplace=True)
            unevaluated_df.index.name = None
        importance_df = pd.concat([evaluated_df, unevaluated_df]).sort_values(by='importance', axis=0)
        # keep features whose importance scores are above threshold or have not had a chance to be calculated
        # only prune up to prune_ratio * n_features features at once
        candidate_features = importance_df[(importance_df['importance'] > prune_threshold) | (importance_df['importance'].isna())].index.tolist()
        removed_features = importance_df[importance_df['importance'] <= prune_threshold].index.tolist()
        candidate_features = candidate_features + removed_features[:len(removed_features) - max(1, int(prune_ratio * n_features))]
        return candidate_features, importance_df

    def fit_score_model(self, model: AbstractModel, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
                        features: Sequence[str], **kwargs) -> Tuple[AbstractModel, float, float]:
        """
        Fits and scores a model. Updates self.time_limit. Returns the fitted model, its score, and time elapsed.
        """
        time_start = time.time()
        X = X[features]
        X_val = None if self.is_bagged else X_val[features]
        model.fit(X=X, y=y, X_val=X_val, y_val=y_val, **kwargs)
        score = model.score_with_oof(y) if self.is_bagged else model.score(X=X_val, y=y_val)
        time_elapsed = time.time() - time_start
        self.time_limit = self.time_limit - time_elapsed
        return model, round(score, 4), round(time_elapsed, 4)
