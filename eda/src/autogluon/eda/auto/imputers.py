import logging

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def fit_imputer(path, df, columns, label, show_all_stages=False, show_leaderboards=True, show_importance=False):
    _df_train = df[~df[label].isna()][columns].reset_index(drop=True)
    (_df_train, _df_val) = train_test_split(_df_train, random_state=0, test_size=0.3)

    predictor = __fit_model(_df_train, path, label)

    print(f'Fitting using the following columns: {sorted(columns)}')
    importance = __get_importance(_df_val, predictor, show_all_stages, show_importance, show_leaderboards)

    # Refit only on significant features
    prev_cols = columns
    columns = [label, *importance[(importance.importance > 0) & (importance.p_value < 0.1)].index.values]
    while columns != prev_cols:
        print(f'  -> {sorted(columns)}')
        predictor = __fit_model(_df_train[columns], path, label)
        importance = __get_importance(_df_val, predictor, show_all_stages, show_importance, show_leaderboards)
        prev_cols = columns
        columns = [label, *importance[(importance.importance > 0) & (importance.p_value < 0.1)].index.values]

    __show_stage_info(_df_val, importance, predictor, show_importance, show_leaderboards)

    return predictor


def __get_importance(_df_val, predictor, show_all_stages, show_importance, show_leaderboards):
    importance = predictor.feature_importance(_df_val.reset_index(drop=True))
    if show_all_stages:
        __show_stage_info(_df_val, importance, predictor, show_importance, show_leaderboards)
    return importance


def __show_stage_info(_df_val, importance, predictor, show_importance, show_leaderboards):
    if show_leaderboards:
        display(predictor.leaderboard(_df_val, silent=True))
    if show_importance:
        display(sns.barplot(data=importance.reset_index(), y='index', x='importance'))
        plt.show()


def __fit_model(_df_train, path, target):
    hyperparameters = {'GBM': {}}
    predictor = TabularPredictor(
        label=target,
        path=path,
        verbosity=0
    ).fit(
        _df_train,
        hyperparameters=hyperparameters,
    )
    return predictor
