from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pyod.models.lof import LOF

from autogluon.eda import AnalysisState
from autogluon.eda.analysis import AnomalyDetector, AnomalyDetectorAnalysis


def test_AnomalyDetector():
    df, cols = _generate_dataset()
    idx_anomaly = [8, 12, 25, 22, 76, 48]
    df.loc[idx_anomaly, cols] = 5
    ad = AnomalyDetector(label="label", n_folds=2, detector_list=[LOF(n_neighbors=5), LOF(n_neighbors=10)])
    anomaly_scores = ad.fit_transform(df)
    assert sorted(anomaly_scores.sort_values(ascending=False).index[: len(idx_anomaly)]) == sorted(idx_anomaly)
    assert ad.predict(pd.DataFrame({"A": [0], "B": [1]}))[0] < 1
    assert ad.predict(pd.DataFrame({"A": [5], "B": [10]}))[0] > 1


def _generate_dataset(seed=0):
    np.random.seed(seed)
    cols = list("AB")
    df = pd.DataFrame(np.random.randint(0, 2, size=(1000, 2)), columns=cols)
    df["label"] = [0, 1] * int(len(df) / 2)
    return df, cols


def test_AnomalyDetector__defaults_init():
    ad = AnomalyDetector("label", some_kwarg="value")

    assert len(ad.detector_list) == 7
    assert len(ad._suod_kwargs["base_estimators"]) == 7
    ad._suod_kwargs.pop("base_estimators")

    assert ad._suod_kwargs["n_jobs"] > 0
    ad._suod_kwargs.pop("n_jobs")
    assert ad._suod_kwargs == {
        "bps_flag": False,
        "combination": "average",
        "some_kwarg": "value",
        "verbose": False,
    }

    assert ad._detectors is None
    assert ad.original_features is None
    assert ad.problem_type == "regression"


def test_AnomalyDetectorAnalysis(monkeypatch):
    call_create_detector = MagicMock()
    call_create_detector.fit_transform.side_effect = lambda x: {
        "train_data_df": "train_df_score",
    }[x]

    call_create_detector.transform.side_effect = lambda x: {
        "test_data_df": "test_df_score",
        "val_data_df": "val_df_score",
    }[x]

    with monkeypatch.context() as m:
        m.setattr(AnomalyDetectorAnalysis, "_create_detector", lambda *args: call_create_detector)
        ad = AnomalyDetectorAnalysis(
            train_data="train_data_df",
            test_data="test_data_df",
            val_data="val_data_df",
            label="label",
            n_folds=3,
        )
        ad.can_handle = MagicMock(return_value=True)
        state = ad.fit()

    assert state == {
        "anomaly_detection": {
            "scores": {"test_data": "test_df_score", "train_data": "train_df_score", "val_data": "val_df_score"}
        }
    }


def test_AnomalyDetectorAnalysis__interpret(monkeypatch):
    call_create_detector = MagicMock()
    call_create_detector.fit_transform.side_effect = lambda x: {
        "train_data_df": "train_df_score",
    }[x]

    call_create_detector.transform.side_effect = lambda x: {
        "test_data_df": "test_df_score",
    }[x]

    with monkeypatch.context() as m:
        m.setattr(AnomalyDetectorAnalysis, "_create_detector", lambda *args: call_create_detector)
        ad = AnomalyDetectorAnalysis(
            train_data="train_data_df",
            test_data="test_data_df",
            label="label",
            store_explainability_data=True,
        )
        ad.can_handle = MagicMock(return_value=True)
        state = ad.fit()

    explain_rows_fns = state.anomaly_detection.pop("explain_rows_fns")
    for ds, fn in explain_rows_fns.items():
        assert fn.args == (
            {"train_data": "train_data_df", "test_data": "test_data_df", "label": "label"},
            call_create_detector,
            ds,
        )

    assert state == {"anomaly_detection": {"scores": {"test_data": "test_df_score", "train_data": "train_df_score"}}}


def test_AnomalyDetectorAnalysis__create_detector():
    a = AnomalyDetectorAnalysis(n_folds=3, some_arg=42)
    ad = a._create_detector(AnalysisState(label="label"))
    assert ad.label == "label"
    assert ad.n_folds == 3
    assert ad._suod_kwargs["some_arg"] == 42


def test_AnomalyDetectorAnalysis__can_handle():
    a = AnomalyDetectorAnalysis()
    assert a.can_handle(None, AnalysisState()) is False

    df_train, cols = _generate_dataset()
    df_test, cols = _generate_dataset()

    assert a.can_handle(None, AnalysisState(train_data=df_train, label="label")) is True
    assert a.can_handle(None, AnalysisState(train_data=df_train, test_data=df_test, label="label")) is True

    _df_train = df_train.copy()
    _df_train.loc[[0, 1, 2], cols] = np.NAN
    _df_test = df_test.copy()
    _df_test.loc[[3, 4, 5], cols] = np.NAN

    assert a.can_handle(None, AnalysisState(train_data=_df_train, test_data=_df_test, label="label")) is False
    assert a.can_handle(None, AnalysisState(train_data=df_train, test_data=_df_test, label="label")) is False


@pytest.mark.parametrize("dataset", ["train_data", "test_data"])
def test_AnomalyDetectorAnalysis__explain_rows_fn(dataset):
    a = AnomalyDetectorAnalysis()
    detector = MagicMock()
    train_data = pd.DataFrame({"train_data": np.arange(4)})
    ds_data = pd.DataFrame({dataset: np.arange(4)})

    kwargs = {} if dataset == "train_data" else {dataset: ds_data}
    args = AnalysisState(train_data=train_data, **kwargs)

    result = a.explain_rows_fn(args, detector=detector, dataset=dataset, dataset_row_ids=[1, 3])

    # np.int64 is required for running tests on Windows
    assert_frame_equal(result["rows"].astype(np.int64), pd.DataFrame({dataset: [1, 3]}, index=[1, 3]).astype(np.int64))
    result.pop("rows")

    assert result["train_data"] is train_data
    result.pop("train_data")

    assert result == {
        "model": detector,
    }
