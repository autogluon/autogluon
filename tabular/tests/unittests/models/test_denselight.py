import logging
import re

import numpy as np
import pandas as pd
import pytest

from autogluon.common.features.types import R_BOOL, R_CATEGORY, R_FLOAT, R_INT
from autogluon.tabular.models.denselight.denselight_model import DenseLightModel
from autogluon.tabular.testing import FitHelper

toy_model_params = {"n_epochs": 2, "patience": 2, "batch_size": 32}


def test_denselight():
    model_cls = DenseLightModel
    model_hyperparameters = toy_model_params

    FitHelper.verify_model(
        model_cls=model_cls,
        model_hyperparameters=model_hyperparameters,
        verify_load_wo_cuda=True,
    )


def test_denselight_auxiliary_params_match_custom_model_tutorial():
    """valid_raw_types should match the custom-model tutorial style (int/float/category/bool)."""
    model = DenseLightModel(problem_type="binary", eval_metric=None)
    aux = model._get_default_auxiliary_params()
    assert set(aux["valid_raw_types"]) == {R_BOOL, R_INT, R_FLOAT, R_CATEGORY}


def _toy_frame(n_rows: int, n_classes: int, seed: int = 0):
    """Small learnable frame: the label is a deterministic function of the features."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f0": rng.normal(size=n_rows),
            "f1": rng.normal(size=n_rows),
            "c0": rng.choice(["a", "b", "c"], n_rows),
        }
    )
    y = pd.Series((X["f0"] * 2 + X["f1"]).rank(method="first").astype(int) % n_classes, name="label")
    return X, y


def _fit_model(X, y, X_val=None, y_val=None, problem_type="multiclass", **params):
    model = DenseLightModel(
        problem_type=problem_type,
        eval_metric=None,
        hyperparameters={"n_epochs": 6, "patience": 6, "batch_size": 32, **params},
    )
    model.fit(X=X, y=y, X_val=X_val, y_val=y_val)
    return model


def test_denselight_early_stopping_restores_batchnorm_statistics():
    """The returned model must actually achieve the best score reported during training.

    Snapshotting `model.parameters()` misses BatchNorm's `running_mean` / `running_var` buffers, so
    the restored model carries best-epoch weights with last-epoch statistics -- a model that was
    never evaluated. The score AutoGluon records is then optimistic and unachievable.

    Read the per-epoch scores out of the training log rather than any attribute, so this asserts
    the same invariant against both the fixed and unfixed implementation.
    """
    X, y = _toy_frame(400, n_classes=2, seed=1)
    X_val, y_val = _toy_frame(200, n_classes=2, seed=2)

    # Capture with an explicit handler: AutoGluon reconfigures logger levels during fit, so
    # caplog's level override does not survive into the training loop.
    records: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("autogluon.tabular.models.denselight._denselight_internal")
    handler = _Collect(level=15)
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(15)
    try:
        model = _fit_model(X, y, X_val, y_val, problem_type="binary", n_epochs=30, patience=30)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    net = model.model.model_
    assert any("running_mean" in name for name, _ in net.named_buffers()), (
        "expected BatchNorm buffers; without use_bn this test proves nothing"
    )

    epoch_scores = [float(m) for m in re.findall(r"DenseLight epoch=\d+ val=(-?\d+\.\d+)", "\n".join(records))]
    assert len(epoch_scores) > 1, f"expected multiple epochs, parsed {epoch_scores}"
    best_reported = max(epoch_scores)

    achieved = model.score(X_val, y_val)
    assert achieved == pytest.approx(best_reported, rel=1e-6, abs=1e-6), (
        f"returned model scores {achieved}, but training reported a best of {best_reported}"
    )


def test_denselight_fits_when_last_batch_has_one_row():
    """`BatchNorm1d` raises in train mode on a single-row batch.

    With batch_size=32 and an internal 80/20 split this is reachable from ordinary row counts, and
    it is a hard failure rather than a degradation.
    """
    batch_size = 32
    n_train = 97  # 97 % 32 == 1
    X, y = _toy_frame(n_train, n_classes=2, seed=3)
    X_val, y_val = _toy_frame(64, n_classes=2, seed=4)
    model = _fit_model(X, y, X_val, y_val, problem_type="binary", batch_size=batch_size, n_epochs=2)
    assert len(model.predict(X_val)) == len(X_val)


def test_denselight_predict_proba_width_when_class_missing_from_train_split():
    """`num_classes` is authoritative; the labels present in the train split are not.

    A rare class absent from a small bagged fold otherwise undersizes the output head and returns
    too few predict_proba columns.
    """
    X, y = _toy_frame(300, n_classes=3, seed=5)
    X_val, y_val = _toy_frame(150, n_classes=3, seed=6)
    # Drop every row of the highest class from the training split only.
    keep = y != 2
    assert (~keep).any(), "test setup must remove at least one row"

    model = DenseLightModel(
        problem_type="multiclass",
        eval_metric=None,
        hyperparameters={"n_epochs": 2, "patience": 2, "batch_size": 32},
    )
    model._num_classes = 3
    model.fit(X=X[keep], y=y[keep], X_val=X_val, y_val=y_val, num_classes=3)

    proba = model.predict_proba(X_val)
    assert proba.shape[1] == 3, f"expected 3 columns for 3 classes, got {proba.shape[1]}"
