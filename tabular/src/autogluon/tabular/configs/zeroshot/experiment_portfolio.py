import copy

# FIXME: DELETE THIS AFTER BENCHMARK. DO NOT PUSH THIS CODE TO MAINLINE.

from .zeroshot_portfolio_2023 import hyperparameter_portfolio_zeroshot_2023

path_weights_tabpfn_mix7_600000 = '/home/ubuntu/tabpfn_weights/TabPFN_mix_7_step_600000.pt'

experiment_portfolio_1 = copy.deepcopy(hyperparameter_portfolio_zeroshot_2023)
experiment_portfolio_2 = copy.deepcopy(hyperparameter_portfolio_zeroshot_2023)
experiment_portfolio_3 = copy.deepcopy(hyperparameter_portfolio_zeroshot_2023)
experiment_portfolio_4 = copy.deepcopy(hyperparameter_portfolio_zeroshot_2023)
experiment_portfolio_5 = copy.deepcopy(hyperparameter_portfolio_zeroshot_2023)
experiment_portfolio_6 = copy.deepcopy(hyperparameter_portfolio_zeroshot_2023)


tabpfn_1 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "ag_args": {"name_suffix": "_c1", "priority": 85},
}

tabpfn_2 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 1, "max_epochs": 30,
    "ag_args": {"name_suffix": "_c2", "priority": 85},
}

tabpfn_3 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 1, "max_epochs": 100,
    "ag_args": {"name_suffix": "_c3", "priority": 85},
}

tabpfn_4 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 4, "max_epochs": 30,
    "ag_args": {"name_suffix": "_c4", "priority": 85},
}

tabpfn_5 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 4, "max_epochs": 100,
    "ag_args": {"name_suffix": "_c5", "priority": 85},
}

tabpfn_4v2 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 4, "max_epochs": 30,
    "ag_args": {"name_suffix": "_c4ns", "priority": 85, "valid_stacker": False},
}

tabpfn_5v2 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 4, "max_epochs": 100,
    "ag_args": {"name_suffix": "_c5ns", "priority": 85, "valid_stacker": False},
}

tabpfn_6 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 16, "max_epochs": 30,
    "ag_args": {"name_suffix": "_c6", "priority": 85},
}

tabpfn_7 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 32, "max_epochs": 30,
    "ag_args": {"name_suffix": "_c7", "priority": 85},
}

tabpfn_8 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 4, "max_epochs": 30,
    "max_samples_query": 2048,
    "max_samples_support": 16382,
    "ag_args": {"name_suffix": "_c8", "priority": 85},
}

tabpfn_9 = {
    "path_weights_classifier": path_weights_tabpfn_mix7_600000,
    "n_ensembles": 4, "max_epochs": 30,
    "max_samples_query": 4096,
    "max_samples_support": 32768,
    "ag_args": {"name_suffix": "_c9", "priority": 85},
}


experiment_portfolio_1["TABPFNMIX"] = [
    tabpfn_1, tabpfn_2,
]

experiment_portfolio_2["TABPFNMIX"] = [
    tabpfn_4, tabpfn_5,
]

experiment_portfolio_3["TABPFNMIX"] = [
    tabpfn_4, tabpfn_5,
]

experiment_portfolio_4["TABPFNMIX"] = [
    tabpfn_4v2, tabpfn_5v2,
]

experiment_portfolio_5["TABPFNMIX"] = [
    tabpfn_8,
]

experiment_portfolio_6["TABPFNMIX"] = [
    tabpfn_9,
]
