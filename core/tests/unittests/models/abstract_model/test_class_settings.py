"""Class settings: process-wide knobs a model class owns, set once rather than per config."""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass

import pytest

from autogluon.core.models import AbstractModel
from autogluon.core.models.abstract._class_settings import ClassSettings


@dataclass(frozen=True)
class _Settings(ClassSettings):
    capacity: int = 1
    verbose: bool = False


class _Model(AbstractModel):
    class_settings_cls = _Settings


class _SubModel(_Model):
    pass


class _PlainModel(AbstractModel):
    pass


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setattr(_Model, "_class_settings", None, raising=False)
    monkeypatch.setattr(_Model, "_class_settings_set", False, raising=False)


def test_class_settings_default_until_set_and_shared_with_subclasses():
    assert _Model.get_class_settings() == _Settings()
    assert _SubModel.get_class_settings() is _Model.get_class_settings(), "a subclass shares the base's settings"
    _SubModel.set_class_settings(capacity=3)
    assert _Model.get_class_settings().capacity == 3
    assert _SubModel.get_class_settings().verbose is False
    assert _PlainModel.get_class_settings() is None


def test_class_settings_reject_unknown_keys_and_undeclared_classes():
    with pytest.raises(ValueError, match="valid settings are"):
        _Model.set_class_settings(capacty=2)
    with pytest.raises(ValueError, match="declares no class settings"):
        _PlainModel.set_class_settings(capacity=2)


def test_class_settings_change_is_logged(caplog):
    logger = logging.getLogger("autogluon.core.models.abstract.abstract_model")
    logger.addHandler(caplog.handler)  # autogluon's loggers do not propagate to the root logger
    try:
        _Model.set_class_settings(capacity=2)
        with caplog.at_level(logging.WARNING, logger=logger.name):
            _Model.set_class_settings(capacity=2)
            assert not caplog.records, "setting the same value again is silent"
            _Model.set_class_settings(capacity=5)
        assert any("class settings change" in record.getMessage() for record in caplog.records)
    finally:
        logger.removeHandler(caplog.handler)


def test_class_settings_snapshot_follows_the_model_into_another_process(tmp_path):
    """A model initialized under some settings re-applies them where its pickle is fit or loaded."""
    _Model.set_class_settings(capacity=4)
    model = _SubModel(path=str(tmp_path), name="m", problem_type="binary", eval_metric="log_loss")
    model.initialize()
    assert model._class_settings_snapshot == {"capacity": 4, "verbose": False}
    saved_path = model.save()

    # Another process starts from defaults.
    _Model._class_settings = None
    _Model._class_settings_set = False
    restored = pickle.loads(pickle.dumps(model))
    restored._apply_class_settings_snapshot()
    assert _Model.get_class_settings().capacity == 4

    _Model._class_settings = None
    _Model._class_settings_set = False
    _SubModel.load(saved_path)
    assert _Model.get_class_settings().capacity == 4

    plain = _PlainModel(path=str(tmp_path / "plain"), name="p", problem_type="binary", eval_metric="log_loss")
    plain.initialize()
    assert plain._class_settings_snapshot is None
