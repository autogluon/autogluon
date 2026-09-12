import os
import sys

import pytest

from autogluon.common.utils import gpu_count, nvutil
from autogluon.common.utils.resource_utils import ResourceManager

torch = pytest.importorskip("torch")

VISIBLE_DEVICE_SETTINGS = [
    None,
    "",
    "0",
    "1",
    "0,1",
    "1,0",
    "-1",
    "0,-1,1",
    "5",
    "0,5,1",
    "0,0",
    " 1 , 0 ",
    "1gpu2,2ampere",
    "GPU-abc",
    "GPU-abc,GPU-abc",
    "GPU-abc,1",
    "MIG-abc",
    "junk",
]


@pytest.mark.parametrize("setting", VISIBLE_DEVICE_SETTINGS)
def test_parse_visible_devices_matches_torch(setting, monkeypatch):
    if setting is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", setting)
    assert gpu_count.parse_visible_devices() == torch.cuda._parse_visible_devices()


def test_uuid_ordinals():
    uuids = ["GPU-aaaa1111", "GPU-aaaa2222", "GPU-bbbb0000"]
    assert gpu_count._uuid_ordinals(["GPU-bbbb"], uuids) == [2]
    assert gpu_count._uuid_ordinals(["GPU-aaaa1", "GPU-bbbb"], uuids) == [0, 2]
    assert gpu_count._uuid_ordinals(["GPU-aaaa"], uuids) == [], "an ambiguous prefix ends the list"
    assert gpu_count._uuid_ordinals(["GPU-bbbb", "GPU-cccc", "GPU-aaaa1"], uuids) == [2], "an unknown id ends the list"
    assert gpu_count._uuid_ordinals(["GPU-bbbb", "GPU-bbbb0"], uuids) == [], "a repeated device gives an empty set"


def test_torch_build_has_cuda_matches_torch():
    assert gpu_count.torch_build_has_cuda() is (torch.version.cuda is not None)


def test_gpu_count_without_torch_matches_torch_in_this_process():
    if not nvutil.cudaInit():
        pytest.skip("NVML is not available")
    nvutil.cudaShutdown()
    count = gpu_count.gpu_count_without_torch(cuda_only=True)
    assert count == torch.cuda.device_count()
    assert ResourceManager.get_gpu_count_torch(cuda_only=True) == torch.cuda.device_count()


def test_get_gpu_count_torch_does_not_import_torch(tmp_path):
    """In a fresh interpreter the count comes from NVML; torch stays unimported."""
    import subprocess

    code = (
        "import sys; from autogluon.common.utils.resource_utils import ResourceManager; "
        "n = ResourceManager.get_gpu_count_torch(); print(n, 'torch' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=os.environ.copy()
    )
    count, torch_imported = out.stdout.split()
    if gpu_count.gpu_count_without_torch() is None:
        pytest.skip("this platform needs torch for the count")
    assert torch_imported == "False"
    assert int(count) == torch.cuda.device_count()


def test_visible_device_memory_matches_torch_totals():
    if not nvutil.cudaInit():
        pytest.skip("NVML is not available")
    nvutil.cudaShutdown()
    memory = gpu_count.visible_device_memory()
    assert memory is not None and len(memory) == torch.cuda.device_count()
    for i, (total, free, used) in enumerate(memory):
        cuda_total = torch.cuda.get_device_properties(i).total_memory
        assert cuda_total <= total <= cuda_total * 1.02, "NVML reports the physical total, CUDA the usable one"
        assert 0 <= used <= total and 0 <= free <= total


def test_cuda_visible_device_count_is_cached_per_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_count.cuda_visible_device_count() == 0
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert gpu_count.cuda_visible_device_count() == 0
    monkeypatch.setattr(gpu_count, "_count_cache", {"": 7})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_count.cuda_visible_device_count() == 7, "a cached value is returned for its visibility setting"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    assert gpu_count.cuda_visible_device_count() == 0, "another setting is computed on its own"
