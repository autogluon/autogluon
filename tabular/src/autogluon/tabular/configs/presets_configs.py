# Dictionary of preset fit() parameter configurations.
tabular_presets_dict = dict(
    # The recommended preset when a GPU is available.
    # State of the art predictive accuracy.
    # Far better than `best_quality` on datasets below 100000 samples, and faster to train.
    # Uses pre-trained tabular foundation models, which add a minimum of 100 MB to the predictor artifact's size.
    # Every model in this preset is free for commercial use; `noncommercial` adds TabPFN-3, whose
    # commercial use requires purchasing a license from Prior Labs: https://docs.priorlabs.ai/models#tabpfn-model-license
    # For best results, use as large of an instance as possible with a GPU (ideally 40+ GB VRAM)
    # Aliases: extreme, extreme_v160, eq
    # GPU STRONGLY RECOMMENDED
    extreme_quality={
        "hyperparameters": "commercial_2026_08_05",
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "num_stack_levels": 0,
        "dynamic_stacking": False,
        "time_limit": 3600,
        "callbacks": [["EarlyStoppingCountCallback", {"patience": [[400, 1], [401, 2], [2000, 2], None]}]],
        "aux_kwargs": {"child_hyperparameters": {"ensemble_size": 40}},
    },
    # Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
    # Recommended for applications that benefit from the best possible model accuracy.
    # Aliases: best
    best_quality={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "hyperparameters": "zeroshot",
        "time_limit": 3600,
    },
    best_quality_v150={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "num_stack_levels": 0,
        "hyperparameters": "zeroshot_2025_12_18_cpu",
        "time_limit": 3600,
        "callbacks": [
            [
                "EarlyStoppingCountCallback",
                {"patience": [[100, 4], [500, 8], [2500, 15], [10000, 40], [100000, 100], None]},
            ]
        ],
    },
    # High predictive accuracy with fast inference. ~8x faster inference and ~8x lower disk usage than `best_quality`.
    # Recommended for applications that require fast inference speed and/or small model size.
    # Aliases: high
    high_quality={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "hyperparameters": "zeroshot",
        "time_limit": 3600,
        "refit_full": True,
        "set_best_to_refit_full": True,
        "save_bag_folds": False,
    },
    high_quality_v150={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "num_stack_levels": 0,
        "hyperparameters": "zeroshot_2025_12_18_cpu",
        "time_limit": 3600,
        "callbacks": [
            [
                "EarlyStoppingCountCallback",
                {"patience": [[100, 4], [500, 8], [2500, 15], [10000, 40], [100000, 100], None]},
            ]
        ],
        "refit_full": True,
        "set_best_to_refit_full": True,
        "save_bag_folds": False,
    },
    # Good predictive accuracy with very fast inference. ~4x faster training, ~8x faster inference and ~8x lower disk usage than `high_quality`.
    # Recommended for applications that require very fast inference speed.
    # Aliases: good
    good_quality={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "hyperparameters": "light",
        "time_limit": 3600,
        "refit_full": True,
        "set_best_to_refit_full": True,
        "save_bag_folds": False,
    },
    # Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than `good_quality`.
    # This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as `good_quality` results in significantly better predictive accuracy with similar inference time.
    # Aliases: medium, medium_quality_faster_train
    medium_quality={"auto_stack": False},
    # Optimizes result immediately for deployment by deleting unused models and removing training artifacts.
    # Often can reduce disk usage by ~2-4x with no negatives to model accuracy or inference speed.
    # This will disable numerous advanced functionality, but has no impact on inference.
    # Recommended for applications where the inner details of AutoGluon's training is not important and there is no intention of manually choosing between the final models.
    # This preset pairs well with the other presets such as `good_quality` to make a very compact final model.
    # Identical to calling `predictor.delete_models(models_to_keep='best', dry_run=False)` and `predictor.save_space()` directly after `fit()`.
    optimize_for_deployment={"keep_only_best": True, "save_space": True},
    # Disables automated feature generation when text features are detected.
    # This is useful to determine how beneficial text features are to the end result, as well as to ensure features are not mistaken for text when they are not.
    ignore_text={
        "_feature_generator_kwargs": {
            "enable_text_ngram_features": False,
            "enable_text_special_features": False,
            "enable_raw_text_features": False,
        }
    },
    ignore_text_ngrams={"_feature_generator_kwargs": {"enable_text_ngram_features": False}},
    # Fit only interpretable models.
    interpretable={
        "auto_stack": False,
        "hyperparameters": "interpretable",
        "feature_generator": "interpretable",
        "fit_weighted_ensemble": False,
        "calibrate": False,
    },
    # ------------------------------------------
    # ------------------------------------------
    # Legacy presets
    # Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
    # Recommended for applications that benefit from the best possible model accuracy.
    best_quality_v082={"auto_stack": True},
    # High predictive accuracy with fast inference. ~10x-200x faster inference and ~10x-200x lower disk usage than `best_quality`.
    # Recommended for applications that require reasonable inference speed and/or model size.
    high_quality_v082={
        "auto_stack": True,
        "refit_full": True,
        "set_best_to_refit_full": True,
        "save_bag_folds": False,
    },
    # Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than `high_quality`.
    # Recommended for applications that require fast inference speed.
    good_quality_v082={
        "auto_stack": True,
        "refit_full": True,
        "set_best_to_refit_full": True,
        "save_bag_folds": False,
        "hyperparameters": "light",
    },
    # ------------------------------------------
    # Experimental presets. Only use these presets if you are ok with unstable and potentially poor performing presets.
    #  Experimental presets can be removed or changed without warning.
    # The v1.5 `extreme_quality`.
    # Aliases: extreme_v150
    extreme_quality_v150={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "num_stack_levels": 0,
        "hyperparameters": "zeroshot_2025_12_18_gpu",
        "time_limit": 3600,
        "callbacks": [
            [
                "EarlyStoppingCountCallback",
                {"patience": [[100, 4], [500, 8], [2500, 15], [10000, 40], [100000, 100], None]},
            ]
        ],
    },
    # [EXPERIMENTAL PRESET] The `noncommercial` preset may be changed or removed without warning.
    # `extreme_quality` plus TabPFN-3, a frontier tabular foundation model created by Prior Labs.
    # Otherwise identical, except that patience is 3 rather than 2 between 401 and 2000 train rows,
    # so the longer foundation-model prefix is reachable there.
    # TabPFN-3 is free for research and internal experimentation; commercial use requires a license
    # or API agreement from Prior Labs: https://docs.priorlabs.ai/models#tabpfn-model-license
    # Aliases: noncommercial_v160
    # GPU STRONGLY RECOMMENDED
    noncommercial={
        "hyperparameters": "noncommercial_2026_08_05",
        "num_bag_folds": 8,
        "num_bag_sets": 1,
        "num_stack_levels": 0,
        "dynamic_stacking": False,
        "time_limit": 3600,
        "callbacks": [["EarlyStoppingCountCallback", {"patience": [[400, 1], [401, 3], [2000, 3], None]}]],
        "aux_kwargs": {"child_hyperparameters": {"ensemble_size": 40}},
    },
    extreme_quality_v140={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "num_bag_sets": 1,
        "_experimental_dynamic_hyperparameters": True,
        "hyperparameters": None,
        "time_limit": 3600,
    },
    # Preset with a portfolio learned from TabArena v0.1: https://tabarena.ai/
    # Uses tabular foundation models: TabPFNv2, TabICL, Mitra
    # Uses deep learning model: TabM
    # Uses tree models: LightGBM, CatBoost, XGBoost
    # Extremely powerful on small datasets with <= 10000 training samples.
    # Requires a GPU for best results.
    tabarena={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "num_bag_sets": 1,
        "num_stack_levels": 0,
        "hyperparameters": "zeroshot_2025_tabfm",
        "time_limit": 3600,
    },
    # DOES NOT SUPPORT GPU.
    experimental_quality_v120={
        "auto_stack": True,
        "dynamic_stacking": "auto",
        "num_bag_sets": 1,
        "hyperparameters": "experimental",
        "fit_strategy": "parallel",
        "num_gpus": 0,
        "time_limit": 3600,
    },
    # ------------------------------------------
    # ------------------------------------------
    # ------------------------------------------
)


# Alias preset name alternatives
tabular_presets_alias = dict(
    extreme="extreme_quality",
    best="best_quality",
    high="high_quality",
    high_quality_fast_inference_only_refit="high_quality",
    good="good_quality",
    good_quality_faster_inference_only_refit="good_quality",
    medium="medium_quality",
    medium_quality_faster_train="medium_quality",
    eq="extreme_quality",
    bq="best_quality",
    hq="high_quality",
    gq="good_quality",
    mq="medium_quality",
    experimental_quality_v140="extreme_quality_v140",
    best_v140="best_quality",
    best_v150="best_quality_v150",
    best_quality_v140="best_quality",
    high_v150="high_quality_v150",
    extreme_v140="extreme_quality_v140",
    extreme_v150="extreme_quality_v150",
    extreme_v160="extreme_quality",
    noncommercial_v160="noncommercial",
)
