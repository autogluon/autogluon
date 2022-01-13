import pytest

from autogluon.tabular.configs.config_helper import ConfigBuilder


def test_with_presets():
    expected_config = dict(presets=['best_quality'])
    acutal_config = ConfigBuilder().with_presets('best_quality').build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().with_presets(['best_quality']).build()
    assert acutal_config == expected_config

    expected_config = dict(presets=['best_quality', 'optimize_for_deployment'])
    acutal_config = ConfigBuilder().with_presets(['best_quality', 'optimize_for_deployment']).build()
    assert acutal_config == expected_config


def test_with_presets_invalid_option():
    with pytest.raises(AssertionError, match=r"unknown1 is not one of the valid presets .*"):
        ConfigBuilder().with_presets('unknown1').build()

    with pytest.raises(AssertionError, match=r"unknown2 is not one of the valid presets .*"):
        ConfigBuilder().with_presets(['unknown2']).build()


def test_with_excluded_model_types():
    expected_config = dict(excluded_model_types=['NN'])
    acutal_config = ConfigBuilder().with_excluded_model_types('NN').build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().with_excluded_model_types(['NN']).build()
    assert acutal_config == expected_config

    expected_config = dict(excluded_model_types=['NN', 'LR'])
    acutal_config = ConfigBuilder().with_excluded_model_types(['NN', 'LR']).build()
    assert acutal_config == expected_config


def test_with_excluded_model_types_invalid_option():
    with pytest.raises(AssertionError, match=r"unknown1 is not one of the valid models .*"):
        ConfigBuilder().with_excluded_model_types('unknown1').build()

    with pytest.raises(AssertionError, match=r"unknown2 is not one of the valid models .*"):
        ConfigBuilder().with_excluded_model_types(['unknown2']).build()


def test_with_included_model_types():
    expected_config = dict(excluded_model_types=['RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'QNN', 'LR', 'FASTAI', 'TRANSF', 'AG_TEXT_NN', 'AG_IMAGE_NN', 'FASTTEXT', 'VW'])
    acutal_config = ConfigBuilder().with_included_model_types('NN').build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().with_included_model_types(['NN']).build()
    assert acutal_config == expected_config

    expected_config = dict(excluded_model_types=['RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'QNN', 'FASTAI', 'TRANSF', 'AG_TEXT_NN', 'AG_IMAGE_NN', 'FASTTEXT', 'VW'])
    acutal_config = ConfigBuilder().with_included_model_types(['NN', 'LR']).build()
    assert acutal_config == expected_config


def test_with_included_model_types_invalid_option():
    with pytest.raises(AssertionError, match=r"unknown1 is not one of the valid models .*"):
        ConfigBuilder().with_included_model_types('unknown1').build()

    with pytest.raises(AssertionError, match=r"unknown2 is not one of the valid models .*"):
        ConfigBuilder().with_included_model_types(['unknown2']).build()


def test_with_time_limit():
    expected_config = dict(time_limit=10)
    acutal_config = ConfigBuilder().with_time_limit(10).build()
    assert acutal_config == expected_config


def test_with_time_limit_invalid_option():
    with pytest.raises(AssertionError, match=r"time_limit must be greater than zero"):
        ConfigBuilder().with_time_limit(-1).build()


def test_with_hyperparameters_str():
    expected_config = dict(hyperparameters='very_light')
    acutal_config = ConfigBuilder().with_hyperparameters('very_light').build()
    assert acutal_config == expected_config


def test_with_hyperparameters_dict():
    expected_config = dict(hyperparameters={'NN': {}})
    acutal_config = ConfigBuilder().with_hyperparameters({'NN': {}}).build()
    assert acutal_config == expected_config


def test_with_hyperparameters__invalid_option():
    with pytest.raises(ValueError, match=r"hyperparameters must be either str: .* or dict with keys of .*"):
        ConfigBuilder().with_hyperparameters(42).build()

    with pytest.raises(AssertionError, match=r"unknown is not one of the valid presets .*"):
        ConfigBuilder().with_hyperparameters('unknown').build()

    with pytest.raises(AssertionError, match=r"The following model types are not recognized: .'unknown'. - use one of the valid models: .*"):
        ConfigBuilder().with_hyperparameters({'unknown': []}).build()
