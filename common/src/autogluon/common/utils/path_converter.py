import os
import platform
from pathlib import Path, PurePosixPath, PureWindowsPath


class PathConverter:
    """Util class to convert a given path to a path to the corresponding OS"""

    @staticmethod
    def _is_windows():
        return platform.system() == "Windows"

    @staticmethod
    def _is_absolute(path: str) -> bool:
        return PureWindowsPath(path).is_absolute() or PurePosixPath(path).is_absolute()

    @staticmethod
    def _validate_path(path: str):
        assert not PathConverter._is_absolute(
            path
        ), "It is ambiguous on how to convert an absolute path. Please provide a relative path instead"

    @staticmethod
    def to_windows(path: str) -> str:
        PathConverter._validate_path(path)
        return str(PathConverter._to_windows(path))

    @staticmethod
    def _to_windows(path: str) -> PureWindowsPath:
        return PureWindowsPath(PurePosixPath(path))

    @staticmethod
    def to_posix(path: str) -> str:
        PathConverter._validate_path(path)
        return str(PathConverter._to_posix(path))

    @staticmethod
    def _to_posix(path: str) -> PurePosixPath:
        return PurePosixPath(PureWindowsPath(path))

    @staticmethod
    def to_current(path: str) -> str:
        return PathConverter.to_windows(path) if PathConverter._is_windows() else PathConverter.to_posix(path)

    @staticmethod
    def os_path_sep() -> str:
        return os.path.sep

    # v0.9 FIXME: Avoid calling this as much as possible
    #  Refactor code so that calling this is not necessary
    @staticmethod
    def to_relative(path: str) -> str:
        if path == "":
            return path
        if not PathConverter._is_absolute(path=path):
            return path
        os_path_sep = PathConverter.os_path_sep()
        path_relative = os.path.relpath(path)
        if path.endswith(os_path_sep):
            if not path_relative.endswith(os_path_sep):
                path_relative += os_path_sep
        return path_relative

    @staticmethod
    def to_absolute(path: str) -> str:
        if path == "":
            return path
        if PathConverter._is_absolute(path=path):
            return path
        path_absolute = str(Path(path).resolve())
        os_path_sep = PathConverter.os_path_sep()
        if path.endswith(os_path_sep):
            if not path_absolute.endswith(os_path_sep):
                path_absolute += os_path_sep
        return path_absolute
