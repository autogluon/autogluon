import pytest

from autogluon.common.utils.resource_utils import RayResourceManager, ResourceManager, get_resource_manager
from autogluon.common.utils.system_info import get_ag_system_info

CLUSTER_RESOURCES = {"CPU": 96.0, "GPU": 8.0, "memory": 512 * 1024**3}


@pytest.fixture
def distributed_mode(monkeypatch):
    """Distributed mode with the cluster lookups stubbed, so no ray cluster is needed."""
    monkeypatch.setenv("AG_DISTRIBUTED_MODE", "1")
    monkeypatch.setattr(
        RayResourceManager,
        "_get_cluster_resources",
        staticmethod(lambda key, default_val=0: CLUSTER_RESOURCES.get(key, default_val)),
    )


def test_get_gpu_count_torch_is_the_local_count(distributed_mode):
    assert get_resource_manager() is RayResourceManager
    assert RayResourceManager.get_gpu_count() == 8
    for cuda_only in (False, True):
        assert RayResourceManager.get_gpu_count_torch(cuda_only=cuda_only) == ResourceManager.get_gpu_count_torch(
            cuda_only=cuda_only
        )


@pytest.mark.parametrize("include_gpu_count", [False, True])
def test_system_info_in_distributed_mode(distributed_mode, include_gpu_count):
    msg = get_ag_system_info(include_gpu_count=include_gpu_count)
    assert "CPU Count:          96" in msg
