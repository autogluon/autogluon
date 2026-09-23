"""`ParallelDistributedFoldFittingStrategy` refuses an S3 model-sync configuration."""

import pytest

from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelDistributedFoldFittingStrategy


@pytest.mark.parametrize("name", ["AG_MODEL_SYNC_PATH", "AG_UTIL_PATH"])
def test_distributed_folding_rejects_s3_sync(monkeypatch, name):
    monkeypatch.setenv("AG_DISTRIBUTED_MODE", "1")
    monkeypatch.setenv(name, "s3://bucket/prefix/")
    # The check runs before any fold-fitting setup, so no bag, data or ray cluster is needed to reach it.
    with pytest.raises(ValueError, match=f"{name} is set"):
        ParallelDistributedFoldFittingStrategy()
