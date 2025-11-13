import pytest

from autogluon.timeseries.trainer import TimeSeriesTrainer
from autogluon.timeseries.trainer.ensemble_composer import EnsembleComposer, validate_ensemble_hyperparameters

from ..common import DUMMY_TS_DATAFRAME


@pytest.fixture(scope="module")
def trainer(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("agts_ensemble_composer_dummy_trainer"))
    trainer = TimeSeriesTrainer(path=path, prediction_length=3)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"Naive": {}, "SeasonalNaive": {}},
    )
    yield trainer


@pytest.mark.parametrize(
    "ensemble_hyperparameters",
    [
        {"GreedyEnsemble": {"ensemble_size": 5}},
        {"GreedyEnsemble": {}, "PerformanceWeightedEnsemble": {}},
        {"PerformanceWeightedEnsemble": {"weight_scheme": "sq"}},
        {"SimpleAverageEnsemble": {}},
        {},
    ],
)
def test_when_ensemble_composer_created_then_can_train_ensembles(tmp_path, trainer, ensemble_hyperparameters):
    """Test that the ensemble composer can create ensembles correctly."""
    ensemble_composer = EnsembleComposer(
        path=trainer.path,
        prediction_length=trainer.prediction_length,
        eval_metric=trainer.eval_metric,
        target=trainer.target,
        quantile_levels=trainer.quantile_levels,
        model_graph=trainer.model_graph,
        ensemble_hyperparameters=ensemble_hyperparameters,
        window_splitter=trainer._get_val_splitter(),
    )
    ensemble_composer.fit(DUMMY_TS_DATAFRAME)

    ensembles = list(ensemble_composer.iter_ensembles())
    assert len(ensembles) == len(ensemble_hyperparameters)

    for layer_ix, _, base_models in ensembles:
        assert layer_ix == 1
        assert len(base_models) >= 1


class TestValidateEnsembleHyperparameters:
    def test_given_valid_hyperparameters_when_validate_called_then_hyperparameters_returned(self):
        """Test validation of valid ensemble hyperparameters dict."""
        hyperparams = {"GreedyEnsemble": {}, "PerformanceWeightedEnsemble": {"some_param": "value"}}
        result = validate_ensemble_hyperparameters(hyperparams)
        assert result == hyperparams

    def test_given_invalid_ensemble_name_when_validate_called_then_error_raised(self):
        """Test validation fails for unknown ensemble names."""
        hyperparams = {"InvalidEnsemble": {}}
        with pytest.raises(ValueError, match="Unknown ensemble type: InvalidEnsemble"):
            validate_ensemble_hyperparameters(hyperparams)

    def test_given_non_dict_input_when_validate_called_then_error_raised(self):
        """Test validation fails for non-dict input."""
        with pytest.raises(ValueError, match="ensemble_hyperparameters must be dict"):
            validate_ensemble_hyperparameters("invalid")

    def test_given_empty_dict_when_validate_called_then_empty_dict_returned(self):
        """Test validation of empty dict."""
        result = validate_ensemble_hyperparameters({})
        assert result == {}
