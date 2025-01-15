import os.path
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from autogluon.common.utils.utils import setup_outputdir


class SetupOutputDirTestCase(unittest.TestCase):
    def test(self):
        # checks that setup_outputdir raises when incorrect type is given
        with self.assertRaises(Exception):
            path = 2.2
            setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)

        # checks that setup_outputdir returns a path AutogluonModels/ag-* when no path is given
        path = None
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
        assert os.path.join("AutogluonModels", "ag") in returned_path

        # checks that setup_outputdir returns the path given as input when given a path of type `str`
        path = tempfile.TemporaryDirectory().name
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
        assert str(Path(returned_path)) == path

        # checks that setup_outputdir returns the path given as input when given a path of type `pathlib.Path`
        path = Path(tempfile.TemporaryDirectory().name)
        returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
        assert str(Path(returned_path)) == str(path)

        # checks that setup_outputdir handles S3 paths correctly
        with patch("os.makedirs") as mock_makedirs:  # Mock os.makedirs to ensure no local directory is created
            path = "s3://test-bucket/test-folder"
            returned_path = setup_outputdir(path, warn_if_exist=True, create_dir=False, path_suffix=None)
            mock_makedirs.assert_not_called()
            self.assertEqual(returned_path, path)


if __name__ == "__main__":
    unittest.main()
