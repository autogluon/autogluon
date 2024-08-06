import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_affected_stacked_overfitting_model_names(leaderboard: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Given a leaderboard from `predictor.leaderboard(test_data)`, return the names of all models that are affected by stacked overfitting and the names of all
    models that are not affected.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        A leaderboard produced by `predictor.leaderboard(test_data)`.
        The leaderboard needs to contain `model` as column.

    Returns
    -------
    non_affected, List of str, names of all non-affected models in the leaderboard.
    affected, List of str, names of all affected models in the leaderboard.
    """

    non_affected = []
    affected = []
    stack_level = 2
    leaderboard_mapping = leaderboard.set_index("model")
    model_to_level_map = leaderboard_mapping["stack_level"].to_dict()
    for model_name in set(leaderboard["model"]):
        # TODO: move away from using names to metadata properties (somehow).
        #   - include something like linear models once we are sure that they cannot leak
        if (
            model_name.startswith("WeightedEnsemble") and model_to_level_map[model_name] <= stack_level
        ) or model_to_level_map[model_name] <= (stack_level - 1):
            non_affected.append(model_name)
        else:
            affected.append(model_name)

    return non_affected, affected


def get_best_val_models(leaderboard: pd.DataFrame) -> Tuple[str, str, bool]:
    """
    Given a leaderboard from `predictor.leaderboard(test_data)`, determine the best model based on validation score that is affected by stacked overfitting,
    the best model that is not affected, and whether any affected models exist at all.

    Parameters
    ----------
    leaderboard : pd.DataFrame
        A leaderboard produced by `predictor.leaderboard(test_data)`.
        The leaderboard needs to contain `score_val` and `model` as columns.

    Returns
    -------
    best_non_affected_model, str, name of the best model that is not affected.
    best_affected_model, str, name of the best model that is affected.
    affected_models_exist, bool, that specifics whether any affected models exist in the given leaderboard.
    """
    non_affected, affected = get_affected_stacked_overfitting_model_names(leaderboard=leaderboard)

    best_non_affected_model = (
        leaderboard[leaderboard["model"].isin(non_affected)]
        .sort_values(by="score_val", ascending=False)
        .iloc[0]
        .loc["model"]
    )

    affected_models_exist = len(affected) > 0
    best_affected_model = None
    if affected_models_exist:
        best_affected_model = (
            leaderboard[leaderboard["model"].isin(affected)]
            .sort_values(by="score_val", ascending=False)
            .iloc[0]
            .loc["model"]
        )

    return best_non_affected_model, best_affected_model, affected_models_exist


def _check_stacked_overfitting_for_models(
    best_non_affected_model: str, best_affected_model: str, leaderboard: pd.DataFrame
) -> bool:
    """
    Determine whether stacked overfitting occurred for the given two models and a leaderboard containing their scores.

    Stacked overfitting occurred, if the validation score of the `best_non_affected_model` is lower than the validation score of the `best_affected_model`
    while the test score of the `best_affected_model` is lower or equal to the test score of `best_non_affected_model`.

    Parameters
    ----------
    best_non_affected_model : str
        Name of the best model, based on validation score, that is not affected by stacked overfitting in principle.
    best_affected_model : str
        Name of the best model, based on validation score, that is affected by stacked overfitting in principle.
    leaderboard : pd.DataFrame
        A leaderboard produced by `predictor.leaderboard(test_data)`.
        The leaderboard needs to contain `score_val` and `model` as columns.

    Returns
    -------
    Bool that is True if stacked overfitting occurred, otherwise False.
    """
    score_non_affected_val = leaderboard.loc[leaderboard["model"] == best_non_affected_model, "score_val"].iloc[0]
    score_non_affected_test = leaderboard.loc[leaderboard["model"] == best_non_affected_model, "score_test"].iloc[0]

    score_affected_val = leaderboard.loc[leaderboard["model"] == best_affected_model, "score_val"].iloc[0]
    score_affected_test = leaderboard.loc[leaderboard["model"] == best_affected_model, "score_test"].iloc[0]

    # l1 worse val score than l2+
    stacked_overfitting = score_non_affected_val < score_affected_val
    # l2+ worse test score than L1
    stacked_overfitting = stacked_overfitting and (score_non_affected_test >= score_affected_test)

    return stacked_overfitting


def check_stacked_overfitting_from_leaderboard(leaderboard: pd.DataFrame) -> bool:
    """
    Determine if stacked overfitting occurred given a leaderboard from `predictor.leaderboard(test_data)`.

    Returns False if there is no model that could have been affected by stacked overfitting in the leaderboard (e.g., no L2 model exists).

    Parameters
    ----------
    leaderboard : pd.DataFrame
        A leaderboard produced by `predictor.leaderboard(test_data)`.
        The leaderboard needs to contain `score_val`, `score_test`, and `model` as columns.

    Returns
    -------
    Bool that is True if stacked overfitting occurred, otherwise False.
    """
    best_non_affected_model, best_affected_model, affected_models_exist = get_best_val_models(leaderboard=leaderboard)

    if not affected_models_exist:
        return False

    return _check_stacked_overfitting_for_models(
        best_non_affected_model=best_non_affected_model,
        best_affected_model=best_affected_model,
        leaderboard=leaderboard,
    )
