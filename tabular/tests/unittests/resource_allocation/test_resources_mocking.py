from autogluon.core.utils import get_cpu_count, get_gpu_count_all


def test_resources_mocking(mock_system_resources_ctx_mgr, mock_num_cpus, mock_num_gpus):
    real_num_cpus = get_cpu_count()
    real_num_gpus = get_gpu_count_all()
    with mock_system_resources_ctx_mgr(num_cpus=mock_num_cpus, num_gpus=mock_num_gpus):
        assert get_cpu_count() == mock_num_cpus
        assert get_gpu_count_all() == mock_num_gpus
    assert get_cpu_count() == real_num_cpus
    assert get_gpu_count_all() == real_num_gpus
