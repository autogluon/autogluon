from autogluon.common.utils.resource_utils import ResourceManager


def test_resources_mocking(mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus):
    real_num_cpus = ResourceManager.get_cpu_count()
    real_num_gpus = ResourceManager.get_gpu_count()
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        assert ResourceManager.get_cpu_count() == mock_num_cpus
        assert ResourceManager.get_gpu_count() == mock_num_gpus
    assert ResourceManager.get_cpu_count() == real_num_cpus
    assert ResourceManager.get_gpu_count() == real_num_gpus
