import pandas as pd
import pytest

from autogluon.tabular import TabularPredictor


@pytest.mark.parametrize("name", ["AG_MODEL_SYNC_PATH", "AG_UTIL_PATH"])
def test_fit_rejects_s3_sync_in_distributed_mode(tmp_path, monkeypatch, name):
    """A leftover S3 model-sync setting fails the fit up front instead of dropping models one by one."""
    monkeypatch.setenv("AG_DISTRIBUTED_MODE", "1")
    monkeypatch.setenv(name, "s3://bucket/prefix/")
    train_data = pd.DataFrame({"a": range(20), "label": [0, 1] * 10})
    with pytest.raises(ValueError, match=f"{name} is set"):
        TabularPredictor(label="label", path=str(tmp_path)).fit(train_data, hyperparameters={"RF": {}})
