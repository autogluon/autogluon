# Noncommercial portfolio: the `extreme_quality` configs plus TabPFN-3, a frontier tabular
# foundation model created by Prior Labs, whose license is not commercially permissive.
# 8 configs. Used by the `noncommercial` preset.
# Fit order is encoded as descending `priority`, which matters because the preset's patience
# callback truncates the fit by model count on small datasets.
hyperparameter_portfolio_noncommercial_2026_08_05 = {
    "TABPFN-3": [{"ag_args": {"priority": -1}}],
    "NORI": [{"ag_args": {"name_suffix": "-30M", "priority": -2}, "model": "nori-30m", "ag.max_rows": 10000}],
    "TABICL": [{"ag_args": {"name_suffix": "v2", "priority": -3}, "ag.max_rows": 100000}],
    "TABDPT-TURBO": [{"ag_args": {"priority": -4}, "ag.max_rows": 100000}],
    "GBM": [
        {
            "ag_args": {"name_prefix": "Prep", "priority": -6},
            "ag_args_ensemble": {"vary_seed_across_folds": True},
            "bagging_fraction": 0.9579806621464,
            "bagging_freq": 1,
            "cat_l2": 0.016204487031,
            "cat_smooth": 0.0014602863645,
            "extra_trees": True,
            "feature_fraction": 0.9895718304666,
            "lambda_l1": 0.3456479366371,
            "lambda_l2": 1.9627316999077,
            "learning_rate": 0.0238015084616,
            "max_cat_to_onehot": 15,
            "min_data_in_leaf": 1,
            "min_data_per_group": 61,
            "num_leaves": 7,
            "ag.model_specific_feature_generator_kwargs": {
                "feature_generators": [
                    [
                        ["GroupByFeatureGenerator", {"max_features": 100}],
                        ["RandomSubsetFeatureCompressionGenerator", {"n_subsets": 50, "random_state": 84}],
                        ["ArithmeticFeatureGenerator", {"max_new_feats": 2000, "random_state": 42}],
                        [
                            [
                                [
                                    "CategoricalInteractionFeatureGenerator",
                                    {"max_new_feats": 500, "passthrough": True, "random_state": 168},
                                ]
                            ],
                            ["OOFTargetEncodingFeatureGenerator", {}],
                        ],
                    ],
                    [["SpearmanFeatureSelector", {"max_features": 2000}]],
                ],
                "passthrough_types": {"invalid_raw_types": ["category", "object"]},
            },
        },
        {
            "bagging_fraction": 0.9688985555289,
            "bagging_freq": 1,
            "cat_l2": 0.556257978498,
            "cat_smooth": 26.9963397207858,
            "extra_trees": True,
            "feature_fraction": 0.721510997519,
            "lambda_l1": 0.918206255573,
            "lambda_l2": 1.6620308841678,
            "learning_rate": 0.0243335590501,
            "max_cat_to_onehot": 63,
            "min_data_in_leaf": 27,
            "min_data_per_group": 20,
            "num_leaves": 16,
            "ag_args": {"name_suffix": "_r8", "priority": -7},
            "ag.min_rows": 50000,
        },
    ],
    "CAT": [{"ag_args": {"priority": -8}, "ag.min_rows": 50000}],
    "REALMLP": [
        {
            "act": "mish",
            "embedding_size": 16,
            "ens_av_before_softmax": False,
            "first_layer_lr_factor": 0.5532758772414772,
            "hidden_sizes": "rectangular",
            "hidden_width": 256,
            "lr": 0.028251710648574225,
            "ls_eps": 0.056278316322438654,
            "ls_eps_sched": "coslog4",
            "max_one_hot_cat_size": 5.0,
            "n_ens": 8,
            "n_epochs": 256,
            "n_hidden_layers": 4,
            "p_drop": 0.4649326053976436,
            "p_drop_sched": "flat_cos",
            "plr_hidden_1": 64,
            "plr_hidden_2": 8,
            "plr_lr_factor": 0.07180754165845323,
            "plr_sigma": 0.10819682427602312,
            "scale_lr_factor": 4.969153878203126,
            "sq_mom": 0.9555568308637205,
            "use_early_stopping": True,
            "use_ls": True,
            "wd": 0.041333041267482,
            "ag_args": {"name_suffix": "_r9", "priority": -9},
            "ag.min_rows": 50000,
        }
    ],
}
