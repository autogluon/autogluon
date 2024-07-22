def test_memory_mocking():
    import psutil

    # Mock patch to guarantee that test does not fail
    # due to memory changing between calls to psutil.
    _mock_vem = psutil.virtual_memory()
    psutil.virtual_memory = lambda: _mock_vem

    # Need to import this after the mock patch above.
    from autogluon.common.utils.resource_utils import ResourceManager

    assert _mock_vem.total == ResourceManager._get_memory_size()
    assert _mock_vem.available == ResourceManager._get_available_virtual_mem()


def test_custom_memory_soft_limit():
    import os

    import psutil

    # Mock patch to guarantee that test does not fail
    # due to memory changing between calls to psutil.
    p = psutil.Process()
    allocated_memory = p.memory_info()
    psutil.Process.memory_info = lambda _: allocated_memory

    # Import after mock patch above.
    from autogluon.common.utils.resource_utils import ResourceManager

    os.environ["AG_MEMORY_LIMIT_IN_GB"] = "48"

    try:
        assert ResourceManager.get_memory_size(format="GB") == 48
        assert (48 * (1024.0**3)) - allocated_memory.rss == ResourceManager.get_available_virtual_mem(format="B")
    finally:
        del os.environ["AG_MEMORY_LIMIT_IN_GB"]
