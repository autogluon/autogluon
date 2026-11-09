"""`InterpretableTabularPredictor` must not write to stdout while validating `fit` kwargs."""

from __future__ import annotations

from autogluon.tabular.predictor import InterpretableTabularPredictor


def test_validate_fit_extra_kwargs_is_silent(capsys, tmp_path):
    """Kwarg validation must not print.

    `_validate_fit_extra_kwargs` runs on every `fit` and `fit_extra` call, after
    `set_logger_verbosity` has been applied, so a bare `print` there dumps the whole
    sanitized kwargs dict to stdout even at `verbosity=0`.
    """
    predictor = InterpretableTabularPredictor(label="label", path=str(tmp_path), verbosity=0)
    capsys.readouterr()  # discard anything the constructor emitted

    kwargs = predictor._validate_fit_extra_kwargs({})

    assert "refit_full" in kwargs  # defaults were populated, so validation actually ran
    assert capsys.readouterr().out == ""
