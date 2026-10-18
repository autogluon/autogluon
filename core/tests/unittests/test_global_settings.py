"""Global settings: process-wide AutoGluon knobs that belong to no single model class."""

from __future__ import annotations

import logging
import pickle

import pytest

import autogluon.core.global_settings as gs
import autogluon.core.models.abstract._shared_weights_registry as registry
from autogluon.core.global_settings import GlobalSettings, get_global_settings, set_global_settings
from autogluon.core.models import AbstractModel


class _Model(AbstractModel):
    pass


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.delenv(registry.CAPACITY_ENV_VAR, raising=False)
    monkeypatch.delenv(registry.LEGACY_CAPACITY_ENV_VAR, raising=False)
    monkeypatch.setattr(gs, "_SETTINGS", GlobalSettings())
    monkeypatch.setattr(gs, "_SET", False)
    registry.set_capacity(None)
    yield
    registry.set_capacity(None)


def _start_fresh_process():
    gs._SETTINGS = GlobalSettings()
    gs._SET = False
    registry.set_capacity(None)


def test_global_settings_default_until_set_and_apply_to_the_registry(monkeypatch):
    assert get_global_settings() == GlobalSettings()
    assert get_global_settings().explicit() == {}
    assert registry.capacity() == registry.DEFAULT_CAPACITY

    assert set_global_settings(shared_weights_capacity=5).shared_weights_capacity == 5
    assert registry.capacity() == 5
    assert get_global_settings().explicit() == {"shared_weights_capacity": 5}

    # None restores the registry default, which the environment variable overrides.
    monkeypatch.setenv(registry.CAPACITY_ENV_VAR, "3")
    set_global_settings(shared_weights_capacity=None)
    assert registry.capacity() == 3
    assert get_global_settings().explicit() == {}


@pytest.mark.parametrize("value", [-1, 1.5, "2", True])
def test_global_settings_reject_invalid_values_and_unknown_keys(value):
    set_global_settings(shared_weights_capacity=4)
    with pytest.raises(ValueError, match="non-negative int or None"):
        set_global_settings(shared_weights_capacity=value)
    with pytest.raises(ValueError, match="valid settings are"):
        set_global_settings(shared_weight_capacity=3)
    assert get_global_settings().shared_weights_capacity == 4
    assert registry.capacity() == 4


def test_global_settings_change_is_logged(caplog):
    logger = logging.getLogger(gs.__name__)
    logger.addHandler(caplog.handler)  # autogluon's loggers do not propagate to the root logger
    try:
        with caplog.at_level(logging.WARNING, logger=logger.name):
            set_global_settings(shared_weights_capacity=2)
            set_global_settings(shared_weights_capacity=2)
            assert not caplog.records, "the first value and setting the same value again are silent"
            set_global_settings(shared_weights_capacity=6)
        assert any("Global settings change" in record.getMessage() for record in caplog.records)
    finally:
        logger.removeHandler(caplog.handler)


def test_global_settings_snapshot_follows_the_model_into_another_process(tmp_path):
    """A model constructed under some settings re-applies them where its pickle is fit or loaded.

    The snapshot is taken at construction, so an uninitialized model shipped to a worker process
    (the fold template of a bag fit in parallel) carries the settings of the launching process.
    """
    unset = _Model(path=str(tmp_path / "unset"), name="u", problem_type="binary", eval_metric="log_loss")
    assert unset._global_settings_snapshot == {}

    set_global_settings(shared_weights_capacity=4)
    model = _Model(path=str(tmp_path), name="m", problem_type="binary", eval_metric="log_loss")
    assert model._global_settings_snapshot == {"shared_weights_capacity": 4}
    saved_path = model.save()
    payload = pickle.dumps(model)

    _start_fresh_process()
    pickle.loads(payload)._apply_settings_snapshots()
    assert registry.capacity() == 4

    _start_fresh_process()
    _Model.load(saved_path)
    assert get_global_settings().shared_weights_capacity == 4
    assert registry.capacity() == 4

    # A model constructed without explicit settings leaves the settings of the process alone.
    unset._apply_settings_snapshots()
    assert registry.capacity() == 4
