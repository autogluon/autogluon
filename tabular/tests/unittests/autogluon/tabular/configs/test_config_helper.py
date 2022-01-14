import pytest

from autogluon.tabular.configs.config_helper import ConfigBuilder


def test_presets():
    expected_config = dict(presets=['best_quality'])
    acutal_config = ConfigBuilder().presets('best_quality').build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().presets(['best_quality']).build()
    assert acutal_config == expected_config

    expected_config = dict(presets=['best_quality', 'optimize_for_deployment'])
    acutal_config = ConfigBuilder().presets(['best_quality', 'optimize_for_deployment']).build()
    assert acutal_config == expected_config


def test_presets_invalid_option():
    with pytest.raises(AssertionError, match=r"The following preset are not recognized: .'unknown1'. - use one of the valid presets: .*"):
        ConfigBuilder().presets('unknown1').build()

    with pytest.raises(AssertionError, match=r"The following preset are not recognized: .'unknown2', 'unknown3'. - use one of the valid presets: .*"):
        ConfigBuilder().presets(['best_quality', 'unknown2', 'unknown3']).build()


def test_excluded_model_types():
    expected_config = dict(excluded_model_types=['NN'])
    acutal_config = ConfigBuilder().excluded_model_types('NN').build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().excluded_model_types(['NN']).build()
    assert acutal_config == expected_config

    expected_config = dict(excluded_model_types=['NN', 'LR'])
    acutal_config = ConfigBuilder().excluded_model_types(['NN', 'LR']).build()
    assert acutal_config == expected_config

    expected_config = dict(excluded_model_types=['NN'])
    acutal_config = ConfigBuilder().excluded_model_types(['NN', 'NN']).build()
    assert acutal_config == expected_config


def test_excluded_model_types_invalid_option():
    with pytest.raises(AssertionError, match=r"unknown1 is not one of the valid models .*"):
        ConfigBuilder().excluded_model_types('unknown1').build()

    with pytest.raises(AssertionError, match=r"unknown2 is not one of the valid models .*"):
        ConfigBuilder().excluded_model_types(['unknown2']).build()


def test_included_model_types():
    expected_config = dict(excluded_model_types=['RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'QNN', 'LR', 'FASTAI', 'TRANSF', 'AG_TEXT_NN', 'AG_IMAGE_NN', 'FASTTEXT', 'VW'])
    acutal_config = ConfigBuilder().included_model_types('NN').build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().included_model_types(['NN']).build()
    assert acutal_config == expected_config

    acutal_config = ConfigBuilder().included_model_types(['NN', 'NN']).build()
    assert acutal_config == expected_config

    expected_config = dict(excluded_model_types=['RF', 'XT', 'KNN', 'GBM', 'CAT', 'XGB', 'QNN', 'FASTAI', 'TRANSF', 'AG_TEXT_NN', 'AG_IMAGE_NN', 'FASTTEXT', 'VW'])
    acutal_config = ConfigBuilder().included_model_types(['NN', 'LR']).build()
    assert acutal_config == expected_config


def test_included_model_types_invalid_option():
    with pytest.raises(AssertionError, match=r"The following model types are not recognized: .'unknown1'. - use one of the valid models: .*"):
        ConfigBuilder().included_model_types('unknown1').build()

    with pytest.raises(AssertionError, match=r"The following model types are not recognized: .'unknown2', 'unknown3'. - use one of the valid models: .*"):
        ConfigBuilder().included_model_types(['RF', 'unknown2', 'unknown3']).build()


def test_time_limit():
    expected_config = dict(time_limit=10)
    acutal_config = ConfigBuilder().time_limit(10).build()
    assert acutal_config == expected_config


def test_time_limit_invalid_option():
    with pytest.raises(AssertionError, match=r"time_limit must be greater than zero"):
        ConfigBuilder().time_limit(-1).build()


def test_hyperparameters_str():
    expected_config = dict(hyperparameters='very_light')
    acutal_config = ConfigBuilder().hyperparameters('very_light').build()
    assert acutal_config == expected_config


def test_hyperparameters_dict():
    expected_config = dict(hyperparameters={'NN': {}})
    acutal_config = ConfigBuilder().hyperparameters({'NN': {}}).build()
    assert acutal_config == expected_config


def test_hyperparameters__invalid_option():
    with pytest.raises(ValueError, match=r"hyperparameters must be either str: .* or dict with keys of .*"):
        ConfigBuilder().hyperparameters(42).build()

    with pytest.raises(AssertionError, match=r"unknown is not one of the valid presets .*"):
        ConfigBuilder().hyperparameters('unknown').build()

    with pytest.raises(AssertionError, match=r"The following model types are not recognized: .'unknown'. - use one of the valid models: .*"):
        ConfigBuilder().hyperparameters({'unknown': []}).build()


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
    assert ConfigBuilder().hyperparameter_tune_kwargs('auto').build() == dict(hyperparameter_tune_kwargs='auto')
    assert ConfigBuilder().hyperparameter_tune_kwargs('random').build() == dict(hyperparameter_tune_kwargs='random')
    assert ConfigBuilder().hyperparameter_tune_kwargs({'props': 42}).build() == dict(hyperparameter_tune_kwargs={'props': 42})
    with pytest.raises(AssertionError, match=r"unknown string must be one of .*"):
        ConfigBuilder().hyperparameter_tune_kwargs('unknown').build()
    with pytest.raises(ValueError, match=r"hyperparameter_tune_kwargs must be either str: .* or dict"):
        ConfigBuilder().hyperparameter_tune_kwargs(42).build()
