
import copy


# Dictionary of preset hyperparameter configurations.
hyperparameter_config_dict = dict(
    # Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
    default={
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}},
        ],
        'XT': [
            {'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}},
        ],
        'KNN': [
            {'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}},
            {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}},
        ],
    },
    # Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
    light={
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
        'RF': [
            {'criterion': 'gini', 'max_depth': 15, 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'max_depth': 15, 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'max_depth': 15, 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}},
        ],
        'XT': [
            {'criterion': 'gini', 'max_depth': 15, 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'entropy', 'max_depth': 15, 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}},
            {'criterion': 'squared_error', 'max_depth': 15, 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}},
        ],
    },
    # Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
    very_light={
        'NN_TORCH': {},
        'GBM': [
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            {},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        'FASTAI': {},
    },
    # Results in extremely quick to train models. Only use this when prototyping, as the model accuracy will be severely reduced.
    toy={
        'NN_TORCH': {'num_epochs': 5},
        'GBM': {'num_boost_round': 10},
        'CAT': {'iterations': 10},
        'XGB': {'n_estimators': 10},
    },
    # Default AutoGluon hyperparameters intended to maximize accuracy in multimodal tabular + text datasets. Requires GPU.
    multimodal={
        'NN_TORCH': {},
        'GBM': [
            {},
            {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
            'GBMLarge',
        ],
        'CAT': {},
        'XGB': {},
        # 'FASTAI': {},  # FastAI gets killed if the dataset is large (400K rows).
        'AG_AUTOMM': {},
        'VW': {},
    },
    # Hyperparameters intended to find an interpretable model which doesn't sacrifice predictive accuracy
    interpretable={
        'IM_RULEFIT': [{'max_rules': 7}, {'max_rules': 12}, {'max_rules': 18}],
        'IM_GREEDYTREE': [{'max_leaf_nodes': 7, 'max_leaf_nodes': 18}],
        'IM_BOOSTEDRULES': [{'n_estimators': 5}, {'n_estimators': 10}],
        'IM_FIGS': [{'max_rules': 6}, {'max_rules': 10}, {'max_rules': 15}],
        'IM_HSTREE': [{'max_rules': 6}, {'max_rules': 12}, {'max_rules': 18}],
    }
)

# default_FTT is experimental
hyperparameter_config_dict['default_FTT'] = {'FT_TRANSFORMER': {}}
hyperparameter_config_dict['default_FTT'].update(hyperparameter_config_dict['default'])


def get_hyperparameter_config_options():
    return list(hyperparameter_config_dict.keys())


def get_hyperparameter_config(config_name):
    config_options = get_hyperparameter_config_options()
    if config_name not in config_options:
        raise ValueError(f'Valid hyperparameter config names are: {config_options}, but \'{config_name}\' was given instead.')
    return copy.deepcopy(hyperparameter_config_dict[config_name])
