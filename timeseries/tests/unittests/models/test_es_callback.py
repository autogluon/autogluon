import pytest

from autogluon.timeseries.predictor import TimeSeriesPredictor
from autogluon.timeseries.models.gluonts.abstract_gluonts import AbstractGluonTSModel
from autogluon.timeseries.models.gluonts.callback import (
    GluonTSEarlyStoppingCallback,
    GluonTSAdaptiveEarlyStoppingCallback
)

from ..common import DUMMY_TS_DATAFRAME


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


def test_early_stopping_patience_passed_to_model(temp_model_path):
    patience = 5
    predictor = TimeSeriesPredictor(path=temp_model_path, early_stopping_patience=patience)
    assert predictor.early_stopping_patience == patience

    # call fit to create trainer
    predictor.fit(DUMMY_TS_DATAFRAME, time_limit=1)

    for m_name in predictor._trainer.get_model_names():
        m = predictor._trainer.load_model(m_name)
        if isinstance(m, AbstractGluonTSModel):
            assert m.early_stopping_patience == patience
            assert len(m.callbacks) == 2

    # setting early_stopping_patience to None to disable early stopping
    predictor = TimeSeriesPredictor(path=temp_model_path, early_stopping_patience=None)
    predictor.fit(DUMMY_TS_DATAFRAME, time_limit=1)

    for m_name in predictor._trainer.get_model_names():
        m = predictor._trainer.load_model(m_name)
        if isinstance(m, AbstractGluonTSModel):
            assert m.early_stopping_patience is None
            assert len(m.callbacks) == 1
