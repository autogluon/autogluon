import logging
from typing import List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def get_affected_stacked_overfitting_model_names(leaderboard: pd.DataFrame) -> Tuple[List[str], List[str]]:
    non_affected = []
    affected = []
    for model_name in set(leaderboard["model"]):
        # TODO: move away from using names to metadata properties (somehow).
        #   - include something like linear models once we are sure that they cannot leak
        if (model_name.startswith("WeightedEnsemble") and model_name.endswith("L2")) or model_name.endswith("L1"):
            non_affected.append(model_name)
        else:
            affected.append(model_name)

    return non_affected, affected


def get_best_val_models(leaderboard: pd.DataFrame) -> Tuple[List[str], List[str], bool]:
    non_affected, affected = get_affected_stacked_overfitting_model_names(leaderboard)

    best_non_affected_model = leaderboard[leaderboard["model"].isin(non_affected)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    affected_models_exist = len(affected) > 0
    best_affected_model = None
    if affected_models_exist:
        best_affected_model = leaderboard[leaderboard["model"].isin(affected)].sort_values(by="score_val", ascending=False).iloc[0].loc["model"]

    return best_non_affected_model, best_affected_model, affected_models_exist


def _check_stacked_overfitting_for_models(best_non_affected_model: List[str], best_affected_model: List[str], leaderboard: pd.DataFrame) -> bool:
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
    best_non_affected_model, best_affected_model, affected_models_exist = get_best_val_models(leaderboard)

    if not affected_models_exist:
        return False

    return _check_stacked_overfitting_for_models(best_non_affected_model, best_affected_model, leaderboard)
