import os
import importlib.util

from typing import Optional


class VersionManager:

    @staticmethod
    def get_ag_version(module: Optional[str] = None) -> str:
        """
        Get AG module version

        Parameters
        ----------
        module: Optional[str], default = None
            Specific module of AG to retrieve version, i.e. "common"
            If not specified, will return the namespace autogluon package version
        """
        ag_root_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", ".."))
        module_version_file_location = os.path.join(ag_root_path, "autogluon", "src", "autogluon", "version.py")
        if module is not None:
            module_version_file_location = os.path.join(ag_root_path, module, "src", "autogluon", module, "version.py")
        module_name = "version"
        # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
        spec = importlib.util.spec_from_file_location(module_name, module_version_file_location)
        version_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(version_module)
        return version_module.__version__
