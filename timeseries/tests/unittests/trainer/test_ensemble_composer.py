import pytest

from autogluon.timeseries.trainer import TimeSeriesTrainer
from autogluon.timeseries.trainer.ensemble_composer import EnsembleComposer, validate_ensemble_hyperparameters

from ..common import DUMMY_TS_DATAFRAME


def test_when_ensemble_composer_created_then_can_train_ensembles(tmp_path):
    """Test that the ensemble composer can create ensembles correctly."""
    trainer = TimeSeriesTrainer(path=str(tmp_path), prediction_length=3)
    trainer.fit(
        train_data=DUMMY_TS_DATAFRAME,
        hyperparameters={"Naive": {}, "SeasonalNaive": {}},
    )

    # Create ensemble composer
    ensemble_composer = EnsembleComposer(
        path=trainer.path,
        prediction_length=trainer.prediction_length,
        eval_metric=trainer.eval_metric,
        target=trainer.target,
        quantile_levels=trainer.quantile_levels,
        model_graph=trainer.model_graph,
        ensemble_hyperparameters={"GreedyEnsemble": {}},
        window_splitter=trainer._get_val_splitter(),
    )

    # Fit ensemble
    ensemble_composer.fit(DUMMY_TS_DATAFRAME)

    # Check that ensemble was created
    ensembles = list(ensemble_composer.iter_ensembles())
    assert len(ensembles) == 1

    layer_ix, model, base_models = ensembles[0]
    assert layer_ix == 1
    assert "WeightedEnsemble" in model.name
    assert len(base_models) >= 1  # At least one base model should be included
    assert all(model in {"Naive", "SeasonalNaive"} for model in base_models)  # Only valid base models


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
