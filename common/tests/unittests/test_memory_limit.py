def test_memory_mocking():
    import psutil
    from autogluon.common.utils.resource_utils import ResourceManager

    assert psutil.virtual_memory().total == ResourceManager._get_memory_size()
    assert psutil.virtual_memory().available == ResourceManager._get_available_virtual_mem()


def test_custom_memory_soft_limit():
    import os

    import psutil

    p = psutil.Process()
    os.environ["AG_MEMORY_LIMIT_IN_GB"] = "48"
    from autogluon.common.utils.resource_utils import ResourceManager

    assert ResourceManager.get_memory_size(format="GB") == 48
    assert (48 * (1024.0**3)) - p.memory_info().rss == ResourceManager.get_available_virtual_mem(format="B")

    del os.environ["AG_MEMORY_LIMIT_IN_GB"]
