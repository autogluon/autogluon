# Dictionary of preset fit() parameter configurations.
tabular_presets_dict = dict(
    # Best predictive accuracy with little consideration to inference time or disk usage. Achieve even better results by specifying a large time_limit value.
    # Recommended for applications that benefit from the best possible model accuracy.
    # Aliases: best
    best_quality={"auto_stack": True},
    # High predictive accuracy with fast inference. ~10x-200x faster inference and ~10x-200x lower disk usage than `best_quality`.
    # Recommended for applications that require reasonable inference speed and/or model size.
    # Aliases: high, high_quality_fast_inference_only_refit
    high_quality={"auto_stack": True, "refit_full": True, "set_best_to_refit_full": True, "_save_bag_folds": False},
    # Good predictive accuracy with very fast inference. ~4x faster inference and ~4x lower disk usage than `high_quality`.
    # Recommended for applications that require fast inference speed.
    # Aliases: good, good_quality_faster_inference_only_refit
    good_quality={"auto_stack": True, "refit_full": True, "set_best_to_refit_full": True, "_save_bag_folds": False, "hyperparameters": "light"},
    # Medium predictive accuracy with very fast inference and very fast training time. ~20x faster training than `good_quality`.
    # This is the default preset in AutoGluon, but should generally only be used for quick prototyping, as `good_quality` results in significantly better predictive accuracy and faster inference time.
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
    ignore_text={"_feature_generator_kwargs": {"enable_text_ngram_features": False, "enable_text_special_features": False, "enable_raw_text_features": False}},
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
    # ------------------------------------------
    # Experimental presets. Only use these presets if you are ok with unstable and potentially poor performing presets.
    #  Experimental presets can be removed or changed without warning.
    # Best quality with an additional FTTransformer model, GPU is recommended.
    experimental_best_quality={"auto_stack": True, "hyperparameters": "default_FTT"},
    # Best quality with an additional FTTransformer and TabPFN model, GPU is recommended.
    #  May have **extremely** slow inference speed, to a potentially unusable degree.
    experimental_extreme_quality={"auto_stack": True, "hyperparameters": "extreme"},
    # Experimental simulated model portfolio.
    # Shown to achieve superior results compared to best_quality on OpenML datasets <5000 rows.
    # Note that runtimes might be much longer than usual with this config.
    experimental_zeroshot_hpo={"auto_stack": True, "hyperparameters": "zeroshot_hpo"},
    experimental_zeroshot_hpo_hybrid={"auto_stack": True, "hyperparameters": "zeroshot_hpo_hybrid"},
    # ------------------------------------------
    # ------------------------------------------
    # ------------------------------------------
    # TODO: Consider HPO-enabled configs if training time doesn't matter but inference latency does.
)


# Alias preset name alternatives
tabular_presets_alias = dict(
    best="best_quality",
    high="high_quality",
    high_quality_fast_inference_only_refit="high_quality",
    good="good_quality",
    good_quality_faster_inference_only_refit="good_quality",
    medium="medium_quality",
    medium_quality_faster_train="medium_quality",
)
