import pytest

from autogluon.common.utils.distribute_utils import DistributedContext


@pytest.fixture(autouse=True)
def _clear_distributed_env(monkeypatch):
    for name in ("AG_DISTRIBUTED_MODE", "AG_MODEL_SYNC_PATH", "AG_UTIL_PATH"):
        monkeypatch.delenv(name, raising=False)


def test_is_distributed_mode(monkeypatch):
    assert not DistributedContext.is_distributed_mode()
    monkeypatch.setenv("AG_DISTRIBUTED_MODE", "1")
    assert DistributedContext.is_distributed_mode()


def test_raise_if_s3_sync_requested_passes_without_sync_vars():
    DistributedContext.raise_if_s3_sync_requested()


@pytest.mark.parametrize("names", [["AG_MODEL_SYNC_PATH"], ["AG_UTIL_PATH"], ["AG_MODEL_SYNC_PATH", "AG_UTIL_PATH"]])
def test_raise_if_s3_sync_requested_names_the_set_vars(monkeypatch, names):
    for name in names:
        monkeypatch.setenv(name, "s3://bucket/prefix/")
    with pytest.raises(ValueError, match="no longer supported") as excinfo:
        DistributedContext.raise_if_s3_sync_requested()
    for name in names:
        assert name in str(excinfo.value)
