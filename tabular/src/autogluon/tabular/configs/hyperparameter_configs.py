import copy

from .zeroshot.zeroshot_portfolio_2023 import hyperparameter_portfolio_zeroshot_2023

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
        "KNN": [
            {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
            {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
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
    zeroshot_hpo={
        "XT": [{"min_samples_leaf": 1, "max_leaf_nodes": 15000, "max_features": 0.5, "ag_args": {"name_suffix": "_r19", "priority": 20}}],
        "RF": [{"min_samples_leaf": 5, "max_leaf_nodes": 50000, "max_features": 0.5, "ag_args": {"name_suffix": "_r5", "priority": 19}}],
        "GBM": [
            {
                "extra_trees": False,
                "feature_fraction": 0.7248284762542815,
                "learning_rate": 0.07947286942946127,
                "min_data_in_leaf": 50,
                "num_leaves": 89,
                "ag_args": {"name_suffix": "_r158", "priority": 18},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.7832570544199176,
                "learning_rate": 0.021720607471727896,
                "min_data_in_leaf": 3,
                "num_leaves": 21,
                "ag_args": {"name_suffix": "_r118", "priority": 17},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.7113010892989156,
                "learning_rate": 0.012535427424259274,
                "min_data_in_leaf": 16,
                "num_leaves": 48,
                "ag_args": {"name_suffix": "_r97", "priority": 16},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.45555769907110816,
                "learning_rate": 0.009591347321206594,
                "min_data_in_leaf": 50,
                "num_leaves": 110,
                "ag_args": {"name_suffix": "_r71", "priority": 15},
            },
            {
                "extra_trees": False,
                "feature_fraction": 0.40979710161022476,
                "learning_rate": 0.008708890211023034,
                "min_data_in_leaf": 3,
                "num_leaves": 80,
                "ag_args": {"name_suffix": "_r111", "priority": 14},
            },
        ],
        "FASTAI": [
            {
                "bs": 1024,
                "emb_drop": 0.6167722379778131,
                "epochs": 44,
                "layers": [200, 100, 50],
                "lr": 0.053440377855629266,
                "ps": 0.48477211305443607,
                "ag_args": {"name_suffix": "_r25", "priority": 13},
            },
            {
                "bs": 1024,
                "emb_drop": 0.6046989241462619,
                "epochs": 48,
                "layers": [200, 100, 50],
                "lr": 0.00775309042164966,
                "ps": 0.09244767444160731,
                "ag_args": {"name_suffix": "_r51", "priority": 12},
            },
            {
                "bs": 512,
                "emb_drop": 0.6557225316526186,
                "epochs": 49,
                "layers": [200, 100],
                "lr": 0.023627682025564638,
                "ps": 0.519566584552178,
                "ag_args": {"name_suffix": "_r82", "priority": 11},
            },
            {
                "bs": 2048,
                "emb_drop": 0.4066210919034579,
                "epochs": 43,
                "layers": [400, 200],
                "lr": 0.0029598312717673434,
                "ps": 0.4378695797438974,
                "ag_args": {"name_suffix": "_r121", "priority": 10},
            },
            {
                "bs": 128,
                "emb_drop": 0.44339037504795686,
                "epochs": 31,
                "layers": [400, 200, 100],
                "lr": 0.008615195908919904,
                "ps": 0.19220253419114286,
                "ag_args": {"name_suffix": "_r145", "priority": 9},
            },
            {
                "bs": 128,
                "emb_drop": 0.12106594798980945,
                "epochs": 38,
                "layers": [200, 100, 50],
                "lr": 0.037991970245029975,
                "ps": 0.33120008492595093,
                "ag_args": {"name_suffix": "_r173", "priority": 8},
            },
            {
                "bs": 128,
                "emb_drop": 0.4599138419358,
                "epochs": 47,
                "layers": [200, 100],
                "lr": 0.03888383281136287,
                "ps": 0.28193673177122863,
                "ag_args": {"name_suffix": "_r128", "priority": 7},
            },
        ],
        "CAT": [
            {"depth": 5, "l2_leaf_reg": 4.774992314058497, "learning_rate": 0.038551267822920274, "ag_args": {"name_suffix": "_r16", "priority": 6}},
            {"depth": 4, "l2_leaf_reg": 1.9950125740798321, "learning_rate": 0.028091050379971633, "ag_args": {"name_suffix": "_r42", "priority": 5}},
            {"depth": 6, "l2_leaf_reg": 1.8298803017644376, "learning_rate": 0.017844259810823604, "ag_args": {"name_suffix": "_r93", "priority": 4}},
            {"depth": 7, "l2_leaf_reg": 4.81099604606794, "learning_rate": 0.019085060180573103, "ag_args": {"name_suffix": "_r44", "priority": 3}},
        ],
    },
    zeroshot_hpo_hybrid={
        "NN_TORCH": {},
        "XT": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
            {"min_samples_leaf": 1, "max_leaf_nodes": 15000, "max_features": 0.5, "ag_args": {"name_suffix": "_r19", "priority": 20}},
        ],
        "RF": [
            {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
            {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
            {"min_samples_leaf": 5, "max_leaf_nodes": 50000, "max_features": 0.5, "ag_args": {"name_suffix": "_r5", "priority": 19}},
        ],
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
            {
                "extra_trees": False,
                "feature_fraction": 0.7248284762542815,
                "learning_rate": 0.07947286942946127,
                "min_data_in_leaf": 50,
                "num_leaves": 89,
                "ag_args": {"name_suffix": "_r158", "priority": 18},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.7832570544199176,
                "learning_rate": 0.021720607471727896,
                "min_data_in_leaf": 3,
                "num_leaves": 21,
                "ag_args": {"name_suffix": "_r118", "priority": 17},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.7113010892989156,
                "learning_rate": 0.012535427424259274,
                "min_data_in_leaf": 16,
                "num_leaves": 48,
                "ag_args": {"name_suffix": "_r97", "priority": 16},
            },
            {
                "extra_trees": True,
                "feature_fraction": 0.45555769907110816,
                "learning_rate": 0.009591347321206594,
                "min_data_in_leaf": 50,
                "num_leaves": 110,
                "ag_args": {"name_suffix": "_r71", "priority": 15},
            },
            {
                "extra_trees": False,
                "feature_fraction": 0.40979710161022476,
                "learning_rate": 0.008708890211023034,
                "min_data_in_leaf": 3,
                "num_leaves": 80,
                "ag_args": {"name_suffix": "_r111", "priority": 14},
            },
        ],
        "XGB": {},
        "FASTAI": [
            {},
            {
                "bs": 1024,
                "emb_drop": 0.6167722379778131,
                "epochs": 44,
                "layers": [200, 100, 50],
                "lr": 0.053440377855629266,
                "ps": 0.48477211305443607,
                "ag_args": {"name_suffix": "_r25", "priority": 13},
            },
            {
                "bs": 1024,
                "emb_drop": 0.6046989241462619,
                "epochs": 48,
                "layers": [200, 100, 50],
                "lr": 0.00775309042164966,
                "ps": 0.09244767444160731,
                "ag_args": {"name_suffix": "_r51", "priority": 12},
            },
            {
                "bs": 512,
                "emb_drop": 0.6557225316526186,
                "epochs": 49,
                "layers": [200, 100],
                "lr": 0.023627682025564638,
                "ps": 0.519566584552178,
                "ag_args": {"name_suffix": "_r82", "priority": 11},
            },
            {
                "bs": 2048,
                "emb_drop": 0.4066210919034579,
                "epochs": 43,
                "layers": [400, 200],
                "lr": 0.0029598312717673434,
                "ps": 0.4378695797438974,
                "ag_args": {"name_suffix": "_r121", "priority": 10},
            },
            {
                "bs": 128,
                "emb_drop": 0.44339037504795686,
                "epochs": 31,
                "layers": [400, 200, 100],
                "lr": 0.008615195908919904,
                "ps": 0.19220253419114286,
                "ag_args": {"name_suffix": "_r145", "priority": 9},
            },
            {
                "bs": 128,
                "emb_drop": 0.12106594798980945,
                "epochs": 38,
                "layers": [200, 100, 50],
                "lr": 0.037991970245029975,
                "ps": 0.33120008492595093,
                "ag_args": {"name_suffix": "_r173", "priority": 8},
            },
            {
                "bs": 128,
                "emb_drop": 0.4599138419358,
                "epochs": 47,
                "layers": [200, 100],
                "lr": 0.03888383281136287,
                "ps": 0.28193673177122863,
                "ag_args": {"name_suffix": "_r128", "priority": 7},
            },
        ],
        "CAT": [
            {},
            {"depth": 5, "l2_leaf_reg": 4.774992314058497, "learning_rate": 0.038551267822920274, "ag_args": {"name_suffix": "_r16", "priority": 6}},
            {"depth": 4, "l2_leaf_reg": 1.9950125740798321, "learning_rate": 0.028091050379971633, "ag_args": {"name_suffix": "_r42", "priority": 5}},
            {"depth": 6, "l2_leaf_reg": 1.8298803017644376, "learning_rate": 0.017844259810823604, "ag_args": {"name_suffix": "_r93", "priority": 4}},
            {"depth": 7, "l2_leaf_reg": 4.81099604606794, "learning_rate": 0.019085060180573103, "ag_args": {"name_suffix": "_r44", "priority": 3}},
        ],
        "KNN": [
            {"weights": "uniform", "ag_args": {"name_suffix": "Unif"}},
            {"weights": "distance", "ag_args": {"name_suffix": "Dist"}},
        ],
    },
    zeroshot=hyperparameter_portfolio_zeroshot_2023,
    zeroshot_2023=hyperparameter_portfolio_zeroshot_2023,
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
