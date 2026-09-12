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
