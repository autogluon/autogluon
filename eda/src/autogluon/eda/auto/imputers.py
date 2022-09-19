import logging
from typing import Union, Dict, Any

from IPython.display import display
from sklearn.model_selection import train_test_split

from autogluon.tabular import TabularPredictor

logger = logging.getLogger(__name__)


def fit_imputer(path, df, columns, label, show_all_stages=False, show_leaderboards=True, show_importance=False, multi_stage=False,
                fig_args: Union[None, Dict[str, Any]] = {}):
    _df_train = df[~df[label].isna()][columns].reset_index(drop=True)
    (_df_train, _df_val) = train_test_split(_df_train, random_state=0, test_size=0.3)

    predictor = __fit_model(_df_train, path, label)

    print(f'Fitting using the following columns: {sorted(columns)}')
    importance = __get_importance(_df_val, predictor, show_all_stages, show_importance, show_leaderboards)

    if multi_stage:
        # Refit only on significant features
        prev_cols = list(columns)
        columns = [label, *importance[(importance.importance > 0) & (importance.p_value < 0.1)].index.values]
        while columns != prev_cols:
            print(f'  -> {sorted(columns)}')
            predictor = __fit_model(_df_train[columns], path, label)
            importance = __get_importance(_df_val, predictor, show_all_stages, show_importance, show_leaderboards, **fig_args)
            prev_cols = columns
            columns = [label, *importance[(importance.importance > 0) & (importance.p_value < 0.1)].index.values]
            if len(columns) < 2:
                break

    __show_stage_info(_df_val, importance, predictor, show_importance, show_leaderboards, **fig_args)

    return predictor


def __get_importance(_df_val, predictor, show_all_stages, show_importance, show_leaderboards, **fig_args):
    importance = predictor.feature_importance(_df_val.reset_index(drop=True))
    if show_all_stages:
        __show_stage_info(_df_val, importance, predictor, show_importance, show_leaderboards, **fig_args)
    return importance


def __show_stage_info(_df_val, importance, predictor, show_importance, show_leaderboards, **fig_args):
    if show_leaderboards:
        display(predictor.leaderboard(_df_val, silent=True))
    if show_importance:

        display(importance[importance.importance>0.0001])
        # fig, ax = plt.subplots(**fig_args)
        # sns.barplot(ax=ax, data=importance.reset_index(), y='index', x='importance')
        # plt.show(fig)


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
