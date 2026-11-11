"""Warn (do not silently misbehave) when an S3 path is passed to ``TabularPredictor``.

S3 paths are not supported for predictor artifacts (see issue #4856): deleting
models does not work and files are repeatedly downloaded during fit. These tests
assert that a warning is emitted for the ``path`` argument of ``__init__`` and
``load`` without training any model, so they stay fast.
"""

from __future__ import annotations

import logging

import pytest

from autogluon.tabular import TabularPredictor

S3_PATH = "s3://fake-bucket/fake-predictor"
WARNING_SUBSTRING = "S3 paths are not supported"


def test_init_warns_on_s3_path(caplog):
    with caplog.at_level(logging.WARNING):
        TabularPredictor(label="label", path=S3_PATH)
    assert any(WARNING_SUBSTRING in record.message for record in caplog.records)


def test_init_no_warning_on_local_path(caplog, tmp_path):
    with caplog.at_level(logging.WARNING):
        TabularPredictor(label="label", path=str(tmp_path))
    assert not any(WARNING_SUBSTRING in record.message for record in caplog.records)


def test_load_warns_on_s3_path(caplog):
    # The warning fires right after the path is resolved, before load reads any
    # file, so loading a nonexistent S3 predictor still triggers it before failing.
    with caplog.at_level(logging.WARNING):
        with pytest.raises(Exception):
            TabularPredictor.load(S3_PATH)
    assert any(WARNING_SUBSTRING in record.message for record in caplog.records)
