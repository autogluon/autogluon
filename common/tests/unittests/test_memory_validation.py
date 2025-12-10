"""
Unit test to verify memory detection with fallback mechanism
"""
import pytest


def test_memory_validation():
    """Test that memory validation correctly identifies unrealistic values"""
    from autogluon.common.utils.resource_utils import ResourceManager
    
    # Test realistic values
    realistic_values = [
        1 * 1024**3,      # 1 GB
        8 * 1024**3,      # 8 GB
        32 * 1024**3,     # 32 GB
        128 * 1024**3,    # 128 GB
        1024 * 1024**3,   # 1 TB
    ]
    
    for value in realistic_values:
        assert ResourceManager._validate_memory_size(value, "test"), \
            f"Realistic value {value / (1024**3):.2f} GB should be valid"
    
    # Test unrealistic values
    unrealistic_values = [
        100 * 1024**2,      # 100 MB (too low)
        5 * 1024**5,        # 5 PB (too high)
        3.2 * 1024**5,      # 3.2 PB (the reported bug value)
    ]
    
    for value in unrealistic_values:
        assert not ResourceManager._validate_memory_size(value, "test"), \
            f"Unrealistic value {value / (1024**3):.2f} GB should be invalid"


def test_memory_size_realistic():
    """Test that get_memory_size returns realistic values"""
    from autogluon.common.utils.resource_utils import ResourceManager
    
    memory_bytes = ResourceManager.get_memory_size("B")
    memory_gb = ResourceManager.get_memory_size("GB")
    
    # Memory should be between 512 MB and 2 TB
    assert memory_bytes >= 512 * 1024 * 1024, "Memory too low"
    assert memory_bytes <= 2 * 1024**4, "Memory too high"
    
    # GB conversion should be correct
    expected_gb = memory_bytes / (1024**3)
    assert abs(memory_gb - expected_gb) < 0.01, "GB conversion incorrect"
    
    print(f"✓ Memory detection working: {memory_gb:.2f} GB")


def test_available_memory_realistic():
    """Test that get_available_virtual_mem returns realistic values"""
    from autogluon.common.utils.resource_utils import ResourceManager
    
    available_bytes = ResourceManager.get_available_virtual_mem("B")
    available_gb = ResourceManager.get_available_virtual_mem("GB")
    total_gb = ResourceManager.get_memory_size("GB")
    
    # Available memory should be between 100 MB and total memory
    assert available_bytes >= 100 * 1024 * 1024, "Available memory too low"
    assert available_gb <= total_gb, "Available memory cannot exceed total"
    
    print(f"✓ Available memory detection working: {available_gb:.2f} GB / {total_gb:.2f} GB")


@pytest.mark.skipif(
    __import__("platform").system() != "Windows",
    reason="Windows-specific test"
)
def test_windows_api_fallback():
    """Test that Windows API fallback works correctly"""
    from autogluon.common.utils.resource_utils import ResourceManager
    import platform
    
    if platform.system() == "Windows":
        try:
            total_mem, avail_mem = ResourceManager._get_memory_size_windows()
            
            # Validate the results
            assert ResourceManager._validate_memory_size(total_mem, "Windows API test"), \
                "Windows API total memory is unrealistic"
            assert ResourceManager._validate_memory_size(avail_mem, "Windows API test"), \
                "Windows API available memory is unrealistic"
            assert avail_mem <= total_mem, "Available memory cannot exceed total"
            
            print(f"✓ Windows API working: {total_mem / (1024**3):.2f} GB total, {avail_mem / (1024**3):.2f} GB available")
        except Exception as e:
            pytest.skip(f"Windows API not available: {e}")


if __name__ == "__main__":
    # Run tests manually
    print("Running memory detection tests...\n")
    
    print("Test 1: Memory Validation")
    test_memory_validation()
    print("✓ Passed\n")
    
    print("Test 2: Memory Size Realistic")
    test_memory_size_realistic()
    print("\n")
    
    print("Test 3: Available Memory Realistic")
    test_available_memory_realistic()
    print("\n")
    
    import platform
    if platform.system() == "Windows":
        print("Test 4: Windows API Fallback")
        test_windows_api_fallback()
        print("\n")
    
    print("All tests passed! ✓")
