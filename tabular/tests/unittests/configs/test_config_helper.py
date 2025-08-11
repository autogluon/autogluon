import numpy as np
import pytest
from sklearn.feature_extraction.text import CountVectorizer

from autogluon.features import TextNgramFeatureGenerator
from autogluon.tabular.configs.config_helper import ConfigBuilder, FeatureGeneratorBuilder
from autogluon.tabular.models import KNNModel
from autogluon.tabular.registry import ag_model_registry


def test_presets():
    expected_config = dict(presets=["best_quality"])
    actual_config = ConfigBuilder().presets("best_quality").build()
    assert actual_config == expected_config

    actual_config = ConfigBuilder().presets(["best_quality"]).build()
    assert actual_config == expected_config

    expected_config = dict(presets=["best_quality", "optimize_for_deployment"])
    actual_config = ConfigBuilder().presets(["best_quality", "optimize_for_deployment"]).build()
    assert actual_config == expected_config

    expected_config = dict(presets={"a": 42})
    actual_config = ConfigBuilder().presets({"a": 42}).build()
    assert actual_config == expected_config


def test_presets_invalid_option():
    with pytest.raises(
        AssertionError,
        match=r"The following presets are not recognized: .'unknown1'. - use one of the valid presets: .*",
    ):
        ConfigBuilder().presets("unknown1").build()

    with pytest.raises(
        AssertionError,
        match=r"The following presets are not recognized: .'unknown2', 'unknown3'. - use one of the valid presets: .*",
    ):
        ConfigBuilder().presets(["best_quality", "unknown2", "unknown3"]).build()


def test_excluded_model_types():
    expected_config = dict(excluded_model_types=["RF"])
    actual_config = ConfigBuilder().excluded_model_types("RF").build()
    assert actual_config == expected_config

    actual_config = ConfigBuilder().excluded_model_types(["RF"]).build()
    assert actual_config == expected_config

    expected_config = dict(excluded_model_types=["LR", "RF"])
    actual_config = ConfigBuilder().excluded_model_types(["RF", "LR"]).build()
    assert actual_config == expected_config

    expected_config = dict(excluded_model_types=["RF"])
    actual_config = ConfigBuilder().excluded_model_types(["RF", "RF"]).build()
    assert actual_config == expected_config


def test_excluded_model_types_invalid_option():
    with pytest.raises(AssertionError, match=r"unknown1 is not one of the valid models .*"):
        ConfigBuilder().excluded_model_types("unknown1").build()

    with pytest.raises(AssertionError, match=r"unknown2 is not one of the valid models .*"):
        ConfigBuilder().excluded_model_types(["unknown2"]).build()


def test_included_model_types():
    model_keys = ag_model_registry.keys
    model_keys_no_rf = [k for k in model_keys if k not in ["RF", "ENS_WEIGHTED", "SIMPLE_ENS_WEIGHTED"]]

    expected_config = dict(excluded_model_types=model_keys_no_rf)
    actual_config = ConfigBuilder().included_model_types("RF").build()
    assert actual_config == expected_config

    actual_config = ConfigBuilder().included_model_types(["RF"]).build()
    assert actual_config == expected_config

    actual_config = ConfigBuilder().included_model_types(["RF", "RF"]).build()
    assert actual_config == expected_config

    model_keys_no_lr = [k for k in model_keys_no_rf if k != "LR"]
    expected_config = dict(excluded_model_types=model_keys_no_lr)
    actual_config = ConfigBuilder().included_model_types(["RF", "LR"]).build()
    assert actual_config == expected_config

    class CustomKNN(KNNModel):
        pass

    expected_config = dict(excluded_model_types=model_keys_no_rf)
    actual_config = ConfigBuilder().included_model_types([CustomKNN, "RF"]).build()
    assert actual_config == expected_config


def test_included_model_types_invalid_option():
    with pytest.raises(
        AssertionError,
        match=r"The following model types are not recognized: .'unknown1'. - use one of the valid models: .*",
    ):
        ConfigBuilder().included_model_types("unknown1").build()

    with pytest.raises(
        AssertionError,
        match=r"The following model types are not recognized: .'unknown2', 'unknown3'. - use one of the valid models: .*",
    ):
        ConfigBuilder().included_model_types(["RF", "unknown2", "unknown3"]).build()


def test_time_limit():
    expected_config = dict(time_limit=10)
    actual_config = ConfigBuilder().time_limit(10).build()
    assert actual_config == expected_config

    expected_config = dict(time_limit=None)
    actual_config = ConfigBuilder().time_limit(None).build()
    assert actual_config == expected_config


def test_time_limit_invalid_option():
    with pytest.raises(AssertionError, match=r"time_limit must be greater than zero"):
        ConfigBuilder().time_limit(-1).build()


def test_hyperparameters_str():
    expected_config = dict(hyperparameters="very_light")
    actual_config = ConfigBuilder().hyperparameters("very_light").build()
    assert actual_config == expected_config


def test_hyperparameters_dict():
    expected_config = dict(hyperparameters={"RF": {}})
    actual_config = ConfigBuilder().hyperparameters({"RF": {}}).build()
    assert actual_config == expected_config

    class CustomKNN(KNNModel):
        pass

    expected_config = dict(hyperparameters={CustomKNN: [{}, {"prop": 42}]})
    actual_config = ConfigBuilder().hyperparameters({CustomKNN: [{}, {"prop": 42}]}).build()
    assert actual_config == expected_config


def test_hyperparameters__invalid_option():
    with pytest.raises(ValueError, match=r"hyperparameters must be either str: .* or dict with keys of .*"):
        ConfigBuilder().hyperparameters(42).build()

    with pytest.raises(AssertionError, match=r"unknown is not one of the valid presets .*"):
        ConfigBuilder().hyperparameters("unknown").build()

    with pytest.raises(
        AssertionError,
        match=r"The following model types are not recognized: .'unknown'. - use one of the valid models: .*",
    ):
        ConfigBuilder().hyperparameters({"unknown": []}).build()


def test_auto_stack():
    assert ConfigBuilder().auto_stack().build() == dict(auto_stack=True)
    assert ConfigBuilder().auto_stack(False).build() == dict(auto_stack=False)


def test_use_bag_holdout():
    assert ConfigBuilder().use_bag_holdout().build() == dict(use_bag_holdout=True)
    assert ConfigBuilder().use_bag_holdout(False).build() == dict(use_bag_holdout=False)


def test_num_bag_folds():
    assert ConfigBuilder().num_bag_folds(0).build() == dict(num_bag_folds=0)
    with pytest.raises(AssertionError, match=r"num_bag_folds must be greater or equal than zero"):
        ConfigBuilder().num_bag_folds(-1).build()


def test_num_bag_sets():
    assert ConfigBuilder().num_bag_sets(1).build() == dict(num_bag_sets=1)
    with pytest.raises(AssertionError, match=r"num_bag_sets must be greater than zero"):
        ConfigBuilder().num_bag_sets(0).build()


def test_num_stack_levels():
    assert ConfigBuilder().num_stack_levels(0).build() == dict(num_stack_levels=0)
    with pytest.raises(AssertionError, match=r"num_stack_levels must be greater or equal than zero"):
        ConfigBuilder().num_stack_levels(-1).build()


def test_holdout_frac():
    assert ConfigBuilder().holdout_frac(0).build() == dict(holdout_frac=0)
    assert ConfigBuilder().holdout_frac(1).build() == dict(holdout_frac=1)
    with pytest.raises(AssertionError, match=r"holdout_frac must be between 0 and 1"):
        ConfigBuilder().holdout_frac(-0.1).build()
    with pytest.raises(AssertionError, match=r"holdout_frac must be between 0 and 1"):
        ConfigBuilder().holdout_frac(1.1).build()


def test_hyperparameter_tune_kwargs():
    assert ConfigBuilder().hyperparameter_tune_kwargs("auto").build() == dict(hyperparameter_tune_kwargs="auto")
    assert ConfigBuilder().hyperparameter_tune_kwargs("random").build() == dict(hyperparameter_tune_kwargs="random")
    assert ConfigBuilder().hyperparameter_tune_kwargs({"props": 42}).build() == dict(
        hyperparameter_tune_kwargs={"props": 42}
    )
    with pytest.raises(AssertionError, match=r"unknown string must be one of .*"):
        ConfigBuilder().hyperparameter_tune_kwargs("unknown").build()
    with pytest.raises(ValueError, match=r"hyperparameter_tune_kwargs must be either str: .* or dict"):
        ConfigBuilder().hyperparameter_tune_kwargs(42).build()


def test_ag_args():
    assert ConfigBuilder().ag_args({"param": 42}).build() == dict(ag_args={"param": 42})


def test_ag_args_fit():
    assert ConfigBuilder().ag_args_fit({"param": 42}).build() == dict(ag_args_fit={"param": 42})


def test_ag_args_ensemble():
    assert ConfigBuilder().ag_args_ensemble({"param": 42}).build() == dict(ag_args_ensemble={"param": 42})


def test_set_best_to_refit_full():
    assert ConfigBuilder().set_best_to_refit_full().build() == dict(set_best_to_refit_full=True)
    assert ConfigBuilder().set_best_to_refit_full(False).build() == dict(set_best_to_refit_full=False)


def test_keep_only_best():
    assert ConfigBuilder().keep_only_best().build() == dict(keep_only_best=True)
    assert ConfigBuilder().keep_only_best(False).build() == dict(keep_only_best=False)


def test_save_space():
    assert ConfigBuilder().save_space().build() == dict(save_space=True)
    assert ConfigBuilder().save_space(False).build() == dict(save_space=False)


def test_calibrate():
    assert ConfigBuilder().calibrate().build() == dict(calibrate=True)
    assert ConfigBuilder().calibrate(False).build() == dict(calibrate=False)


def test_use_bag_holdout():
    assert ConfigBuilder().use_bag_holdout().build() == dict(use_bag_holdout=True)
    assert ConfigBuilder().use_bag_holdout(False).build() == dict(use_bag_holdout=False)


def test_refit_full():
    assert ConfigBuilder().refit_full().build() == dict(refit_full=True)
    assert ConfigBuilder().refit_full(False).build() == dict(refit_full=False)
    assert ConfigBuilder().refit_full("best").build() == dict(refit_full="best")


def test_feature_generator():
    vectorizer = CountVectorizer(min_df=7, ngram_range=(2, 3), max_features=11, dtype=np.uint8)

    config = (
        ConfigBuilder()
        .feature_generator()
        .enable_numeric_features()
        .enable_categorical_features()
        .enable_datetime_features()
        .enable_text_special_features()
        .enable_text_ngram_features()
        .enable_raw_text_features()
        .enable_vision_features()
        .vectorizer(vectorizer)
        .text_ngram_params({"vectorizer_strategy": "both"})
        .build()
        .build()
    )

    assert config["feature_generator"].enable_numeric_features is True
    assert config["feature_generator"].enable_categorical_features is True
    assert config["feature_generator"].enable_datetime_features is True
    assert config["feature_generator"].enable_text_special_features is True
    assert config["feature_generator"].enable_text_ngram_features is True
    assert config["feature_generator"].enable_raw_text_features is True
    assert config["feature_generator"].enable_vision_features is True

    text_gen = None
    generators_classes = []
    for gl in config["feature_generator"].generators:
        for g in gl:
            if isinstance(g, TextNgramFeatureGenerator):
                text_gen = g
            generators_classes.append(g.__class__.__name__)
    print(generators_classes)
    assert str(text_gen.vectorizer_default_raw) == str(vectorizer)
    assert text_gen.vectorizer_strategy == "both"
    assert sorted(list(set(generators_classes))) == [
        "AsTypeFeatureGenerator",
        "CategoryFeatureGenerator",
        "DatetimeFeatureGenerator",
        "DropDuplicatesFeatureGenerator",
        "DropUniqueFeatureGenerator",
        "FillNaFeatureGenerator",
        "IdentityFeatureGenerator",
        "IsNanFeatureGenerator",
        "TextNgramFeatureGenerator",
        "TextSpecialFeatureGenerator",
    ]


def test_feature_generator_2():
    config = (
        ConfigBuilder()
        .feature_generator()
        .enable_numeric_features(False)
        .enable_categorical_features(False)
        .enable_datetime_features(False)
        .enable_text_special_features(False)
        .enable_text_ngram_features(False)
        .enable_raw_text_features(False)
        .enable_vision_features(False)
        .build()
        .build()
    )

    assert config["feature_generator"].enable_numeric_features is False
    assert config["feature_generator"].enable_categorical_features is False
    assert config["feature_generator"].enable_datetime_features is False
    assert config["feature_generator"].enable_text_special_features is False
    assert config["feature_generator"].enable_text_ngram_features is False
    assert config["feature_generator"].enable_raw_text_features is False
    assert config["feature_generator"].enable_vision_features is False

    text_gen = None
    generators_classes = []
    for gl in config["feature_generator"].generators:
        for g in gl:
            if isinstance(g, TextNgramFeatureGenerator):
                text_gen = g
            generators_classes.append(g.__class__.__name__)
    assert text_gen is None
    assert sorted(list(set(generators_classes))) == [
        "AsTypeFeatureGenerator",
        "DropDuplicatesFeatureGenerator",
        "DropUniqueFeatureGenerator",
        "FillNaFeatureGenerator",
    ]


def test_feature_generator_builder_standalone():
    vectorizer = CountVectorizer(min_df=7, ngram_range=(2, 3), max_features=11, dtype=np.uint8)

    generator = (
        FeatureGeneratorBuilder()
        .enable_numeric_features()
        .enable_categorical_features()
        .enable_datetime_features()
        .enable_text_special_features()
        .enable_text_ngram_features()
        .enable_raw_text_features()
        .enable_vision_features()
        .vectorizer(vectorizer)
        .text_ngram_params({"vectorizer_strategy": "both"})
        .build()
    )
    assert generator.enable_numeric_features is True
    assert generator.enable_categorical_features is True
    assert generator.enable_datetime_features is True
    assert generator.enable_text_special_features is True
    assert generator.enable_text_ngram_features is True
    assert generator.enable_raw_text_features is True
    assert generator.enable_vision_features is True
