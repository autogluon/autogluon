import logging

import numpy as np
import pandas as pd

from autogluon.core.constants import PROBLEM_TYPES_CLASSIFICATION

logger = logging.getLogger(__name__)


def sample_bins_uniformly(y_pred_proba: pd.DataFrame, df_indexes):
    """
    Takes predictive probabilities from y_pred_proba and finds the minimum
    class count then samples minimum class count from every class with
    rows with indexes in df_indexes, where minimum class count is determined
    by dividing y_pred_proba into classes based on their max class predictive prob
    and then each class is counted and the lowest number is the number that is sampled
    from each class.

    Parameters:
    y_pred_proba: Predicted probabilities for multi-class problem
    df_indexes: The indices of y_pred_proba should be taken into consideration when
        sampling evenly. These indices should correspond to the indices in
        y_pred_proba.

    Returns:
    pd.Series of indices that were selected by sample
    """
    test_pseudo_indices = pd.Series(data=False, index=y_pred_proba.index)
    if y_pred_proba[df_indexes].empty:
        return test_pseudo_indices
    else:
        pred_idxmax = y_pred_proba[df_indexes].idxmax(axis=1)
        class_value_counts = pred_idxmax.value_counts()
        min_count = class_value_counts.min()
        class_keys = list(class_value_counts.keys())

    if len(class_keys) < 1:
        return test_pseudo_indices

    logging.log(15, f"Taking {min_count} rows from the following classes: {class_keys}")

    new_test_pseudo_indices = None
    for k in class_keys:
        class_k_pseudo_idxes = pred_idxmax == k
        selected_rows = class_k_pseudo_idxes[class_k_pseudo_idxes].sample(min_count)

        if new_test_pseudo_indices is None:
            new_test_pseudo_indices = selected_rows.index
        else:
            new_test_pseudo_indices = new_test_pseudo_indices.append(selected_rows.index)

    test_pseudo_indices.loc[new_test_pseudo_indices] = True

    return test_pseudo_indices


def filter_pseudo(
    y_pred_proba_og,
    problem_type,
    min_proportion_prob: float = 0.05,
    max_proportion_prob: float = 0.6,
    threshold: float = 0.95,
    proportion_sample: float = 0.3,
):
    """
    Takes in the predicted probabilities of the model (y_pred_proba_og) and chooses the indices that meet
    a criteria to incorporate into training data. Criteria is determined by problem_type.
    If multiclass or binary will choose all rows with max prob over threshold. For regression
    chooses 30% of the labeled data randomly. This filter is used pseudo labeled data.

    Parameters:
    -----------
    y_pred_proba_og: The predicted probabilities from the current best model. If problem is
        'binary' or 'multiclass' then it's Panda series of predictive probs, if it's 'regression'
        then it's a scalar. Binary probs should be set to multiclass.
    min_proportion_prob: Minimum proportion of indices in y_pred_proba_og to select. The filter
        threshold will be automatically adjusted until at least min_proportion_prob of the predictions
        in y_pred_proba_og pass the filter. This ensures we return at least min_proportion_prob of the
        pseudolabeled data to augment the training set in pseudolabeling.
    max_proportion_prob: Maximum proportion of indices in y_pred_proba_og to select. The filter threshold
        will be automatically adjusted until at most max_proportion_prob of the predictions in y_pred_proba_og
        pass the filter. This ensures we return at most max_proportion_prob of the pseudolabeled data to augment
        the training set in pseudolabeling.
    threshold: This filter will only return those indices of y_pred_proba_og where the probability
        of the most likely class exceeds the given threshold value.
    proportion_sample: When problem_type is regression this is percent of pseudo data
        to incorporate into train. Rows selected randomly.

    Returns:
    --------
    pd.Series of indices that met pseudo labeling requirements
    """
    if problem_type in PROBLEM_TYPES_CLASSIFICATION:
        y_pred_proba_max = y_pred_proba_og.max(axis=1)
        curr_threshold = threshold
        curr_percentage = (y_pred_proba_max >= curr_threshold).mean()
        num_rows = len(y_pred_proba_max)

        if curr_percentage > max_proportion_prob or curr_percentage < min_proportion_prob:
            if curr_percentage > max_proportion_prob:
                num_rows_threshold = max(np.ceil(max_proportion_prob * num_rows), 1)
            else:
                num_rows_threshold = max(np.ceil(min_proportion_prob * num_rows), 1)
            curr_threshold = y_pred_proba_max.sort_values(ascending=False).iloc[int(num_rows_threshold) - 1]

        test_pseudo_indices = y_pred_proba_max >= curr_threshold
        test_pseudo_indices = sample_bins_uniformly(y_pred_proba=y_pred_proba_og, df_indexes=test_pseudo_indices)
    else:
        test_pseudo_indices = pd.Series(data=False, index=y_pred_proba_og.index)
        test_pseudo_indices_true = test_pseudo_indices.sample(frac=proportion_sample, random_state=0)
        test_pseudo_indices[test_pseudo_indices_true.index] = True

    test_pseudo_indices = test_pseudo_indices[test_pseudo_indices]

    return test_pseudo_indices


def filter_ensemble_pseudo(predictor, unlabeled_data: pd.DataFrame, num_models: int = 5):
    """
    Uses top num_models to predict on unlabeled data then filters the ensemble model
    predicted data and returns indices of row that meet a metric. If problem is multiclass
    or binary, will take top num_models predictive probabilities for unlabeled data then
    averages them and selects row with max predictive probability above a threshold. For
    regression, will take top num_models and calculate the variance in their predictions
    and select the rows with the least variance among all model predictions.

    Parameters:
    -----------
    predictor: Fitted TabularPredictor
    unlabeled_data: Unlabeled data for top k models to predict on
    num_models: Number of top models to ensemble

    Returns:
    -------
    pd.Series of indices that met pseudo labeling requirements
    """
    original_k = num_models
    leaderboard = predictor._trainer.leaderboard()
    num_models = len(leaderboard)

    if num_models < 2:
        raise Exception("Ensemble pseudo labeling was enabled, but only one model was trained")

    if num_models != num_models:
        logging.warning(f"Ensemble pseudo labeling expected {original_k}, but only {num_models} fit.")

    if predictor.problem_type in PROBLEM_TYPES_CLASSIFICATION:
        return filter_ensemble_classification(
            predictor=predictor, unlabeled_data=unlabeled_data, leaderboard=leaderboard, num_models=num_models
        )
    else:
        return filter_pseudo_std_regression(
            predictor=predictor, unlabeled_data=unlabeled_data, leaderboard=leaderboard, num_models=num_models
        )


def filter_pseudo_std_regression(
    predictor,
    unlabeled_data: pd.DataFrame,
    num_models,
    leaderboard,
    lower_bound: float = -0.25,
    upper_bound: float = 0.25,
):
    """
    Predicts on unlabeled_data using the top num_models. Then gets standard deviation of each
    row's predictions across the top num_models and the standard deviation across all rows of standard
    deviations of the top num_models. Calculates z-score using top num_models predictions standard
    deviation minus the mean of the top num_models standard deviation divided by the standard deviation of
    the top num_model predictions across all rows. The indices of all top num_model predictions who's z-score
    falls within lower_bound and upper_bound will be returned.

    Parameters:
    -----------
    predictor: Fitted TabularPredictor
    unlabeled_data: Unlabeled data for top k models to predict on
    leaderboard: pd.DataFrame of leaderboard of models in AutoGluon based on validation score
    num_models: Number of top models to ensemble
    lower_bound: Lower threshold that z-score needs to exceed in order to
        incorporate
    upper_bound: Upper threshold that z-score needs to be below to incorporate

    Returns:
    --------
    pd.Series of indices that met pseudo labeling requirements
    """
    top_k_models_list = leaderboard.head(num_models)["model"]
    top_k_preds = None

    for model in top_k_models_list:
        y_test_pred = predictor.predict(data=unlabeled_data, model=model)

        if model == top_k_models_list[0]:
            top_k_preds = y_test_pred
        else:
            top_k_preds = pd.concat([top_k_preds, y_test_pred], axis=1)

    # Todo: This doesn't make total sense, because weighted ensemble
    # will ensemble the top models and weigh them according to validation score
    # maybe should just force prediction using weighted ensemble instead of
    # uniform across top k?
    top_k_avg_preds = top_k_preds.mean(axis=1)
    top_k_preds = top_k_preds.to_numpy()
    preds_sd = pd.Series(data=np.std(top_k_preds, axis=1), index=unlabeled_data.index)
    preds_z_score = (preds_sd - preds_sd.mean()) / preds_sd.std()
    df_filtered = preds_z_score.between(lower_bound, upper_bound)

    return df_filtered[df_filtered], top_k_avg_preds


def filter_ensemble_classification(
    predictor, unlabeled_data: pd.DataFrame, leaderboard, num_models, threshold: float = 0.95
):
    """
    Calculates predictive probabilities of unlabeled data by predicting with top num_models
    then averages pre-row over predictions from top num_models and selects rows where confidence
    (predicted probability of the most likely class) is above threshold. Then samples minimum
    bin count from all bins, where bins are rows of averaged predictions with the same peak
    predicted probability class.

    Parameters:
    -----------
    predictor: Fitted TabularPredictor
    unlabeled_data: Unlabeled data for top k models to predict on
    leaderboard: pd.DataFrame of leaderboard of models in AutoGluon based on validation score
    num_models: Number of top models to ensemble
    threshold: The predictive probability a row must exceed in order to be
        selected

    Returns:
    --------
    pd.Series of indices that met pseudo labeling requirements
    """
    top_k_model_names = leaderboard.head(num_models)["model"]

    y_pred_proba_ensemble = None
    for model_name in top_k_model_names:
        y_pred_proba_curr_model = predictor.predict_proba(data=unlabeled_data, model=model_name)

        if y_pred_proba_ensemble is None:
            y_pred_proba_ensemble = y_pred_proba_curr_model
        else:
            y_pred_proba_ensemble += y_pred_proba_curr_model

    # Todo: This doesn't make total sense, because weighted ensemble
    # will ensemble the top models and weigh them according to validation score
    # maybe should just force prediction using weighted ensemble instead of
    # uniform across top k?
    y_pred_proba_ensemble /= num_models
    y_max_prob = y_pred_proba_ensemble.max(axis=1)
    pseudo_indexes = y_max_prob >= threshold
    y_pred_ensemble = y_pred_proba_ensemble.idxmax(axis=1)

    test_pseudo_indices = sample_bins_uniformly(y_pred_proba=y_pred_proba_ensemble, df_indexes=pseudo_indexes)

    return test_pseudo_indices[test_pseudo_indices], y_pred_proba_ensemble, y_pred_ensemble


def assert_pseudo_column_match(X: pd.DataFrame, X_pseudo: pd.DataFrame):
    """
    Raises an AssertionError if X and X_pseudo don't share the same columns.
    Useful to call prior to concatenating the data together to avoid unexpected behavior.

    Parameters
    ----------
    X: pd.DataFrame
        The original training data
    X_pseudo: pd.DataFrame
        Additional training data with pseudo-labelled targets
    """
    if set(X.columns) != set(X_pseudo.columns):
        X_unique_cols = sorted(set(X.columns).difference(X_pseudo.columns))
        X_pseudo_unique_cols = sorted(set(X_pseudo.columns).difference(X.columns))
        raise AssertionError(
            f"X and X_pseudo columns are mismatched!\n"
            f"\tUnexpected Columns in X_pseudo: {X_pseudo_unique_cols}\n"
            f"\t   Missing Columns in X_pseudo: {X_unique_cols}"
        )
