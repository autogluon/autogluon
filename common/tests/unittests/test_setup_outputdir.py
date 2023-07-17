import os.path
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
