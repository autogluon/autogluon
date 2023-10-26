import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_non_leaking_and_leaking_model_names(leaderboard: pd.DataFrame) -> Tuple[List[str], List[str]]:
    non_leaking = []
    leaking = []
    for model_name in set(leaderboard["model"]):
        # TODO: move away from using names to metadata properties (somehow).
        if model_name.startswith("WeightedEnsemble") and model_name.endswith("L2"):
            non_leaking.append(model_name)
        else:
            leaking.append(model_name)

    return non_leaking, leaking


def get_best_val_models(leaderboard: pd.DataFrame) -> Tuple[List[str], List[str], bool]:
    non_leaking_names, leaking_names = get_non_leaking_and_leaking_model_names(leaderboard)

    best_non_leaking_model = leaderboard[leaderboard["model"].isin(non_leaking_names)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    leaking_models_exist = len(leaking_names) > 0
    best_leaking_model = None
    if leaking_models_exist:
        best_leaking_model = leaderboard[leaderboard["model"].isin(leaking_names)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    return best_non_leaking_model, best_leaking_model, leaking_models_exist


def _check_stacked_overfitting_for_models(best_non_leaking_model: List[str], best_leaking_model: List[str], leaderboard: pd.DataFrame) -> bool:
    score_non_leaking_oof = leaderboard.loc[leaderboard["model"] == best_non_leaking_model, "score_val"].iloc[0]
    score_non_leaking_test = leaderboard.loc[leaderboard["model"] == best_non_leaking_model, "score_test"].iloc[0]

    score_leaking_oof = leaderboard.loc[leaderboard["model"] == best_leaking_model, "score_val"].iloc[0]
    score_leaking_test = leaderboard.loc[leaderboard["model"] == best_leaking_model, "score_test"].iloc[0]

    # l1 worse val score than l2+
    stacked_overfitting = score_non_leaking_oof < score_leaking_oof
    # l2+ worse test score than L1
    stacked_overfitting = stacked_overfitting and (score_non_leaking_test >= score_leaking_test)

    return stacked_overfitting


def check_stacked_overfitting_from_leaderboard(leaderboard: pd.DataFrame) -> bool:
    best_non_leaking_model, best_leaking_model, leaking_models_exist = get_best_val_models(leaderboard)

    if not leaking_models_exist:
        return False

    return _check_stacked_overfitting_for_models(best_non_leaking_model, best_leaking_model, leaderboard)
