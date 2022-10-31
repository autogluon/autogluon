import pytest

from autogluon.timeseries import MXNET_INSTALLED
if not MXNET_INSTALLED:
    pytest.skip(allow_module_level=True)

from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.models.gluonts.mx.callback import (
    GluonTSAdaptiveEarlyStoppingCallback,
    GluonTSEarlyStoppingCallback,
)
from autogluon.timeseries.predictor import TimeSeriesPredictor

from ....common import DUMMY_TS_DATAFRAME
from ..test_gluonts import TESTABLE_MX_MODELS


@pytest.mark.parametrize(
    "patience, best_round, best_loss, epoch_no, epoch_loss, to_continue",
    [
        (10, 0, 1, 1, 0.9, True),
        (10, 0, 1, 9, 1, True),
        (10, 0, 1, 10, 1, False),
        (10, 10, 1, 11, 0.9, True),
        (10, 10, 1, 20, 1, False),
        (20, 10, 1, 29, 0.9, True),
        (20, 10, 1, 30, 0.9, True),
        (20, 10, 1, 30, 1, False),
    ],
)
def test_simple_early_stopping(patience, best_round, best_loss, epoch_no, epoch_loss, to_continue):
    es = GluonTSEarlyStoppingCallback(patience)
    es.best_round = best_round
    es.best_loss = best_loss
    assert es.on_validation_epoch_end(epoch_no, epoch_loss) is to_continue


@pytest.mark.parametrize(
    "best_round, best_loss, epoch_no, epoch_loss, to_continue",
    [
        (0, 1, 1, 0.9, True),
        (0, 1, 9, 1, True),
        (0, 1, 10, 1, False),
        (10, 1, 11, 0.9, True),
        (10, 1, 20, 1, True),
        (10, 1, 23, 0.9, True),
        (10, 1, 23, 1, False),
        (100, 1, 120, 1, True),
        (100, 1, 140, 1, False),
    ],
)
def test_adaptive_early_stopping(best_round, best_loss, epoch_no, epoch_loss, to_continue):
    es = GluonTSAdaptiveEarlyStoppingCallback()
    es.best_round = best_round
    es.best_loss = best_loss
    es.patience = es.es._update_patience(best_round)
    assert es.on_validation_epoch_end(epoch_no, epoch_loss) is to_continue


@pytest.mark.parametrize(
    "best_round, patience", [(0, 10), (10, 13), (100, 40), (1000, 310), (10000, 3010), (100000, 10000)]
)
def test_adaptive_early_stopping_update_patience(best_round, patience):
    es = GluonTSAdaptiveEarlyStoppingCallback()
    assert es.es._update_patience(best_round) == patience


@pytest.mark.parametrize("model_class", TESTABLE_MX_MODELS)
def test_model_save_load_with_adaptive_es(model_class, temp_model_path):
    model = model_class(
        path=temp_model_path,
        freq="H",
        quantile_levels=[0.1, 0.9],
        hyperparameters={
            "epochs": 1,
        },
    )
    model.fit(
        train_data=DUMMY_TS_DATAFRAME,
    )

    model.callbacks.append(GluonTSAdaptiveEarlyStoppingCallback())
    assert any(isinstance(c, GluonTSAdaptiveEarlyStoppingCallback) for c in model.callbacks)
    model.save()

    loaded_model = model.__class__.load(path=model.path)
    assert any(isinstance(c, GluonTSAdaptiveEarlyStoppingCallback) for c in loaded_model.callbacks)
    assert model.gluonts_estimator_class is loaded_model.gluonts_estimator_class
    assert loaded_model.gts_predictor == model.gts_predictor


def test_early_stopping_patience_used_in_hp(temp_model_path):
    patience = 5

    hps = {
        "SimpleFeedForwardMXNet": {
            "epochs": 5,
            "num_batches_per_epoch": 10,
            "context_length": 5,
            "early_stopping_patience": patience,
        },
        "MQCNNMXNet": {
            "epochs": 5,
            "num_batches_per_epoch": 10,
            "context_length": 5,
            "early_stopping_patience": patience,
        },
        "DeepARMXNet": {
            "epochs": 5,
            "num_batches_per_epoch": 10,
            "context_length": 5,
            "early_stopping_patience": patience,
        },
    }

    predictor = TimeSeriesPredictor(path=temp_model_path)

    # call fit to create trainer
    predictor.fit(DUMMY_TS_DATAFRAME, time_limit=1, hyperparameters=hps)
    for m_name in predictor._trainer.get_model_names():
        model = predictor._trainer.load_model(m_name)
        if isinstance(model, AbstractGluonTSModel):
            assert any(isinstance(c, GluonTSEarlyStoppingCallback) for c in model.callbacks)


def test_early_stopping_patience_not_used_in_hp(temp_model_path):

    hps = {
        "SimpleFeedForwardMXNet": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
        "MQCNNMXNet": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
        "DeepARMXNet": {"epochs": 5, "num_batches_per_epoch": 10, "context_length": 5},
    }

    predictor = TimeSeriesPredictor(path=temp_model_path, hyperparameters=hps)

    # call fit to create trainer
    predictor.fit(DUMMY_TS_DATAFRAME, time_limit=1)

    for m_name in predictor._trainer.get_model_names():
        model = predictor._trainer.load_model(m_name)
        if isinstance(model, AbstractGluonTSModel):
            assert not any(isinstance(c, GluonTSEarlyStoppingCallback) for c in model.callbacks)
