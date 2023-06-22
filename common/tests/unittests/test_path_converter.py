from unittest.mock import patch

import pytest

from autogluon.common.utils.path_converter import PathConverter


@pytest.mark.parametrize("og_path, expected_path", [("dummy", "dummy"), ("dummy/foo", "dummy\\foo")])
def test_to_windows(og_path, expected_path):
    assert PathConverter.to_windows(og_path) == expected_path


@pytest.mark.parametrize(
    "og_path, expected_path",
    [
        ("dummy", "dummy"),
        ("dummy\\foo", "dummy/foo"),
    ],
)
def test_to_posix(og_path, expected_path):
    assert PathConverter.to_posix(og_path) == expected_path


@pytest.mark.parametrize(
    "og_path, expected_path, mock_system",
    [
        ("dummy", "dummy", "Windows"),
        ("dummy/foo", "dummy\\foo", "Windows"),
        ("dummy", "dummy", "Linux"),
        ("dummy\\foo", "dummy/foo", "Linux"),
    ],
)
def test_to_current(og_path, expected_path, mock_system):
    with patch("platform.system", return_value=mock_system):
        assert PathConverter.to_current(og_path) == expected_path


@pytest.mark.parametrize("path", [("/"), ("/home/ubuntu/foo"), ("c:\\User"), ("C:\\User"), ("C:\\")])
def test_should_raise_on_absolute_path(path):
    with pytest.raises(AssertionError):
        PathConverter.to_windows(path)
        PathConverter.to_posix(path)
