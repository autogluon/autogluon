
import copy


# Dictionary of preset hyperparameter configurations.
hyperparameter_config_dict = dict(
    # Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
    default={
        'NN': {},
        'GBM': {},
        'CAT': {},
        'RF': {},
        'XT': {},
        'KNN': {},
        'custom': ['GBM'],
    },
    # Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
    light={
        'NN': {},
        'GBM': {},
        'CAT': {},
        'RF': {'max_depth': 15},
        'XT': {'max_depth': 15},
        'custom': ['GBM'],
    },
    # Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
    very_light={
        'NN': {},
        'GBM': {},
        'CAT': {},
    },
    # Results in extremely quick to train models. Only use this when prototyping, as the model accuracy will be severely reduced.
    toy={
        'NN': {'num_epochs': 10},
        'GBM': {'num_boost_round': 10},
        'CAT': {'iterations': 10},
    }
)


def get_hyperparameter_config_options():
    return list(hyperparameter_config_dict.keys())


def get_hyperparameter_config(config_name):
    config_options = get_hyperparameter_config_options()
    if config_name not in config_options:
        raise ValueError(f'Valid hyperparameter config names are: {config_options}, but \'{config_name}\' was given instead.')
    return copy.deepcopy(hyperparameter_config_dict[config_name])
