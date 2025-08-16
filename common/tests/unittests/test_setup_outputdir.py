from __future__ import annotations

import os.path
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from autogluon.common.utils.utils import DEFAULT_BASE_PATH, setup_outputdir


class SetupOutputDirTestCase(unittest.TestCase):
    def test_os_path(self):
        # checks that setup_outputdir raises when incorrect type is given
        with pytest.raises(Exception):
            path = 2.2
            setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)

        # checks that setup_outputdir returns a path AutogluonModels/ag-* when no path is given
        path = None
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
        assert os.path.join(DEFAULT_BASE_PATH, "ag") in returned_path

        # checks that setup_outputdir returns a path CustomPath/ag-* when base path is given
        path = None
        returned_path = setup_outputdir(
            path,
            warn_if_exist=True,
            create_dir=False,
            path_suffix=None,
            default_base_path="CustomPath",
        )
        assert os.path.join("CustomPath", "ag") in returned_path

        # checks that setup_outputdir returns the path given as input when given a path of type `str`
        path = tempfile.TemporaryDirectory().name
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
        assert str(Path(returned_path)) == path

        # checks that setup_outputdir returns the path given as input when given a path of type `pathlib.Path`
        path = Path(tempfile.TemporaryDirectory().name)
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
        assert str(Path(returned_path)) == str(path)

        # checks behavior of path_suffix logic
        path = tempfile.TemporaryDirectory().name
        path_suffix = f"my_subdir{os.path.sep}"
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=path_suffix)
        assert not returned_path.endswith(os.path.sep)
        assert "my_subdir" in returned_path

    def test_s3_path(self):
        path = "s3://test-bucket/test-folder"
        # checks no local dir is created
        with patch("os.makedirs") as mock_makedirs:
            returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=True, path_suffix=None)
            mock_makedirs.assert_not_called()
            assert returned_path == path

        # checks behavior of path_suffix logic
        path_suffix = "my_subdir/"
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=path_suffix)
        assert not returned_path.endswith("/")
        assert "my_subdir" in returned_path


if __name__ == "__main__":
    unittest.main()
