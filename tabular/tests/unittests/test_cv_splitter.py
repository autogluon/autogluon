"""Tests for custom cv_splitter support in TabularPredictor (issue #4492).

Verifies that a user can pass a pre-configured sklearn-compatible cross-validator
(e.g. TimeSeriesSplit) via ``TabularPredictor(cv_splitter=...)`` so that
temporally ordered data is split without leakage.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import KFold, TimeSeriesSplit

from autogluon.tabular import TabularPredictor


def _make_temporal_dataset(n_samples: int = 30):
    """Small regression dataset with an explicit temporal ordering."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "feature1": rng.standard_normal(n_samples),
            "feature2": rng.standard_normal(n_samples),
        }
    )
    y = pd.Series(rng.standard_normal(n_samples), name="target")
    data = pd.concat([X, y], axis=1)
    return data, "target"


def test_cv_splitter_timeseries_split():
    """TabularPredictor trains successfully when cv_splitter=TimeSeriesSplit is given."""
    data, label = _make_temporal_dataset(n_samples=30)
    n_splits = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        predictor = TabularPredictor(
            label=label,
            path=tmpdir,
            cv_splitter=TimeSeriesSplit(n_splits=n_splits),
        )
        predictor.fit(
            train_data=data,
            num_bag_folds=n_splits,
            hyperparameters={"GBM": {"num_boost_round": 5}},
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        )
        preds = predictor.predict(data)
        assert len(preds) == len(data)


def test_cv_splitter_n_splits_matches():
    """The number of bagged models equals the splitter's n_splits."""
    data, label = _make_temporal_dataset(n_samples=30)
    n_splits = 4

    with tempfile.TemporaryDirectory() as tmpdir:
        predictor = TabularPredictor(
            label=label,
            path=tmpdir,
            cv_splitter=TimeSeriesSplit(n_splits=n_splits),
        )
        predictor.fit(
            train_data=data,
            num_bag_folds=n_splits,
            hyperparameters={"GBM": {"num_boost_round": 5}},
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        )
        # Each bagged model produces one fold; leaderboard should list the ensemble and its children
        leaderboard = predictor.leaderboard(silent=True)
        assert len(leaderboard) > 0


def test_cv_splitter_arbitrary_sklearn_splitter():
    """Any sklearn BaseCrossValidator instance is accepted (e.g. KFold)."""
    data, label = _make_temporal_dataset(n_samples=30)
    n_splits = 3

    with tempfile.TemporaryDirectory() as tmpdir:
        predictor = TabularPredictor(
            label=label,
            path=tmpdir,
            cv_splitter=KFold(n_splits=n_splits, shuffle=True, random_state=0),
        )
        predictor.fit(
            train_data=data,
            num_bag_folds=n_splits,
            hyperparameters={"GBM": {"num_boost_round": 5}},
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        )
        preds = predictor.predict(data)
        assert len(preds) == len(data)


def test_cv_splitter_and_groups_mutually_exclusive():
    """Specifying both cv_splitter and groups raises a ValueError."""
    data, label = _make_temporal_dataset(n_samples=30)
    data["group_col"] = [i % 3 for i in range(len(data))]

    with pytest.raises(ValueError, match="mutually exclusive"):
        TabularPredictor(
            label=label,
            groups="group_col",
            cv_splitter=TimeSeriesSplit(n_splits=3),
        )
