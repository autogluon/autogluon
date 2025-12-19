import copy

from .zeroshot.zeroshot_portfolio_2023 import hyperparameter_portfolio_zeroshot_2023
from .zeroshot.zeroshot_portfolio_2025 import hyperparameter_portfolio_zeroshot_2025_small
from .zeroshot.zeroshot_portfolio_cpu_2025_12_18 import hyperparameter_portfolio_zeroshot_cpu_2025_12_18
from .zeroshot.zeroshot_portfolio_gpu_2025_12_18 import hyperparameter_portfolio_zeroshot_gpu_2025_12_18

# Dictionary of preset hyperparameter configurations.
hyperparameter_config_dict = dict(
    # Default AutoGluon hyperparameters intended to maximize accuracy without significant regard to inference time or disk usage.
    default={
        "NN_TORCH": {},
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ],
        "CAT": {},
        "XGB": {},
        "FASTAI": {},
        "RF": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
        ],
        "XT": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
        ],
    },
    # Results in smaller models. Generally will make inference speed much faster and disk usage much lower, but with worse accuracy.
    light={
        "NN_TORCH": {},
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ],
        "CAT": {},
        "XGB": {},
        "FASTAI": {},
        "RF": [
            {"criterion": "gini", "max_depth": 15, "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "max_depth": 15, "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "max_depth": 15, "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
        ],
        "XT": [
            {"criterion": "gini", "max_depth": 15, "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "max_depth": 15, "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "max_depth": 15, "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
        ],
    },
    # Results in much smaller models. Behaves similarly to 'light', but in many cases with over 10x less disk usage and a further reduction in accuracy.
    very_light={
        "NN_TORCH": {},
        "GBM": [
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ],
        "CAT": {},
        "XGB": {},
        "FASTAI": {},
    },
    # Results in extremely quick to train models. Only use this when prototyping, as the model accuracy will be severely reduced.
    toy={
        "NN_TORCH": {"num_epochs": 5},
        "GBM": {"num_boost_round": 10},
        "CAT": {"iterations": 10},
        "XGB": {"n_estimators": 10},
    },
    # Default AutoGluon hyperparameters intended to maximize accuracy in multimodal tabular + text datasets. Requires GPU.
    multimodal={
        "NN_TORCH": {},
        "GBM": [
            {},
            {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
            {
                "learning_rate": 0.03,
                "num_leaves": 128,
                "feature_fraction": 0.9,
                "min_data_in_leaf": 3,
                "ag_args": {"name_suffix": "Large", "priority": 0, "hyperparameter_tune_kwargs": None},
            },
        ],
        "CAT": {},
        "XGB": {},
        # 'FASTAI': {},  # FastAI gets killed if the dataset is large (400K rows).
        "AG_AUTOMM": {},
    },
    # Hyperparameters intended to find an interpretable model which doesn't sacrifice predictive accuracy
    interpretable={
        "IM_RULEFIT": [{"max_rules": 7}, {"max_rules": 12}, {"max_rules": 18}],
        "IM_FIGS": [{"max_rules": 6}, {"max_rules": 10}, {"max_rules": 15}],
        # Note: Below are commented out because they are not meaningfully interpretable via the existing API
        # 'IM_GREEDYTREE': [{'max_leaf_nodes': 7, 'max_leaf_nodes': 18}],
        # 'IM_BOOSTEDRULES': [{'n_estimators': 5}, {'n_estimators': 10}],
        # 'IM_HSTREE': [{'max_rules': 6}, {'max_rules': 12}, {'max_rules': 18}],
    },
    zeroshot=hyperparameter_portfolio_zeroshot_2023,
    zeroshot_2023=hyperparameter_portfolio_zeroshot_2023,
    zeroshot_2025_tabfm=hyperparameter_portfolio_zeroshot_2025_small,
    zeroshot_2025_12_18_gpu=hyperparameter_portfolio_zeroshot_gpu_2025_12_18,
    zeroshot_2025_12_18_cpu=hyperparameter_portfolio_zeroshot_cpu_2025_12_18,
)

tabpfnmix_default = {
    "model_path_classifier": "autogluon/tabpfn-mix-1.0-classifier",
    "model_path_regressor": "autogluon/tabpfn-mix-1.0-regressor",
    "n_ensembles": 1,
    "max_epochs": 30,
    "ag.sample_rows_val": 5000,  # Beyond 5k val rows fine-tuning becomes very slow
    "ag.max_rows": 50000,  # Beyond 50k rows, the time taken is longer than most users would like (hours), while the model is very weak at this size
    "ag_args": {"name_suffix": "_v1"},
}

hyperparameter_config_dict["experimental_2024"] = {"TABPFNMIX": tabpfnmix_default}
hyperparameter_config_dict["experimental_2024"].update(hyperparameter_config_dict["zeroshot_2023"])
hyperparameter_config_dict["experimental"] = hyperparameter_config_dict["experimental_2024"]

def get_hyperparameter_config_options():
    return list(hyperparameter_config_dict.keys())


def get_hyperparameter_config(config_name):
    config_options = get_hyperparameter_config_options()
    if config_name not in config_options:
        raise ValueError(f"Valid hyperparameter config names are: {config_options}, but '{config_name}' was given instead.")
    return copy.deepcopy(hyperparameter_config_dict[config_name])
