from __future__ import annotations

import os
import os.path
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

from autogluon.common.utils.utils import (
    DEFAULT_BASE_PATH,
    DEFAULT_BASE_PATH_ENV_VAR,
    get_default_base_path,
    setup_outputdir,
)


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

    def test_default_base_path_env_var(self):
        # `AG_DEFAULT_BASE_PATH` redirects auto-generated paths, so callers (e.g. test suites) can
        # keep predictor artifacts out of the working directory without passing `path` everywhere.
        prev = os.environ.get(DEFAULT_BASE_PATH_ENV_VAR)
        try:
            os.environ.pop(DEFAULT_BASE_PATH_ENV_VAR, None)
            assert get_default_base_path() == DEFAULT_BASE_PATH

            with tempfile.TemporaryDirectory() as tmp_dir:
                os.environ[DEFAULT_BASE_PATH_ENV_VAR] = tmp_dir
                assert get_default_base_path() == tmp_dir

                returned_path = setup_outputdir(None, warn_if_exist=True, create_dir=False, path_suffix=None)
                assert returned_path.startswith(os.path.realpath(tmp_dir))
                assert os.path.join(tmp_dir, "ag") in returned_path

                # an explicit `default_base_path` still wins over the env var
                returned_path = setup_outputdir(
                    None,
                    warn_if_exist=True,
                    create_dir=False,
                    path_suffix=None,
                    default_base_path="CustomPath",
                )
                assert os.path.join("CustomPath", "ag") in returned_path

                # an explicit `path` is unaffected by the env var
                explicit = os.path.join(tmp_dir, "explicit")
                returned_path = setup_outputdir(explicit, warn_if_exist=True, create_dir=False, path_suffix=None)
                assert str(Path(returned_path)) == explicit

            # an empty value falls back to the default rather than resolving to ""
            os.environ[DEFAULT_BASE_PATH_ENV_VAR] = ""
            assert get_default_base_path() == DEFAULT_BASE_PATH
        finally:
            if prev is None:
                os.environ.pop(DEFAULT_BASE_PATH_ENV_VAR, None)
            else:
                os.environ[DEFAULT_BASE_PATH_ENV_VAR] = prev

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
