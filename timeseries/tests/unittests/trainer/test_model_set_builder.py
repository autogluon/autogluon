import pytest

from autogluon.common import space
from autogluon.timeseries.models import ChronosModel, DeepARModel
from autogluon.timeseries.models.abstract.abstract_timeseries_model import AbstractTimeSeriesModel
from autogluon.timeseries.trainer.model_set_builder import HyperparameterBuilder, TrainableModelSetBuilder

HP_TEST_CASES = [
    (
        {  # input hyperparameters
            "Chronos": {"a": 1, "b": 2},
            "DeepAR": [{"a": 1, "b": 2}],
        },
        2,  # expected number of distinct models in output hyperparameter spec
        2,  # expected number of total models
    ),
    (
        {
            "Chronos": {"a": 1, "b": 2},
            "DeepAR": [{"a": 1, "b": 2}, {"a": 3, "b": 5}],
        },
        2,
        3,
    ),
    (
        {
            "Chronos": [{"a": 1, "b": 2}],
            "DeepAR": [{"a": 1, "b": 2}],
        },
        2,
        2,
    ),
    (
        {
            "Chronos": {"a": 1, "b": 2},
        },
        1,
        1,
    ),
    (
        {
            ChronosModel: {"a": 1, "b": 2},
        },
        1,
        1,
    ),
    (
        {
            ChronosModel: [{"a": 1, "b": 2}, {"a": 3, "b": 5}],
        },
        1,
        2,
    ),
    (
        {
            DeepARModel: {"a": 1, "b": 2},
        },
        1,
        1,
    ),
    (
        {
            "DeepAR": {"a": 1, "b": 2},
            ChronosModel: [{"a": 3, "b": 5}],
        },
        2,
        2,
    ),
]


@pytest.fixture()
def model_set_builder():
    yield TrainableModelSetBuilder(
        path=None,
        freq="H",
        prediction_length=5,
        eval_metric="MASE",
        target="target",
        quantile_levels=[0.1, 0.5, 0.9],
        covariate_metadata=None,
        multi_window=False,
    )


@pytest.mark.parametrize("hyperparameter_spec, expected_num_specs, expected_num_models", HP_TEST_CASES)
def test_when_hp_builder_called_then_hyperparameters_built_correctly(
    hyperparameter_spec, expected_num_specs, expected_num_models
):
    hps = HyperparameterBuilder(
        hyperparameters=hyperparameter_spec,
        hyperparameter_tune=False,
        excluded_model_types=[],
    ).get_hyperparameters()

    assert len(hps) == expected_num_specs
    for k, v in hps.items():
        assert isinstance(v, list)
        assert isinstance(k, (str, type))


@pytest.mark.parametrize("hyperparameter_spec, expected_num_specs, expected_num_models", HP_TEST_CASES)
def test_when_hp_builder_called_with_tune_but_no_spaces_then_error_is_raised(
    hyperparameter_spec, expected_num_specs, expected_num_models
):
    with pytest.raises(ValueError, match="no model contains a hyperparameter search space"):
        HyperparameterBuilder(
            hyperparameters=hyperparameter_spec,
            hyperparameter_tune=True,
            excluded_model_types=[],
        ).get_hyperparameters()


def test_when_hp_builder_called_with_no_tune_but_has_spaces_then_error_is_raised():
    spec = {ChronosModel: {"a": space.Int(3, 5)}}

    with pytest.raises(ValueError, match="hyperparameters must have fixed values"):
        HyperparameterBuilder(
            hyperparameters=spec,
            hyperparameter_tune=False,
            excluded_model_types=[],
        ).get_hyperparameters()


@pytest.mark.parametrize("hyperparameter_spec, expected_num_specs, expected_num_models", HP_TEST_CASES)
def test_when_model_set_builder_called_then_hyperparameters_built_correctly(
    model_set_builder, hyperparameter_spec, expected_num_specs, expected_num_models
):
    model_set = model_set_builder.get_model_set(
        hyperparameters=hyperparameter_spec,
        hyperparameter_tune=False,
        excluded_model_types=[],
    )

    assert len(model_set) == expected_num_models
    assert all(isinstance(m, AbstractTimeSeriesModel) for m in model_set)


def test_when_non_model_class_provided_to_model_set_builder_then_error_is_raised(model_set_builder):
    class DummyModel:
        pass

    with pytest.raises(ValueError, match="Unknown model"):
        model_set_builder.get_model_set(
            hyperparameters={DummyModel: {"a": 1, "b": 2}},
            hyperparameter_tune=False,
            excluded_model_types=[],
        )
