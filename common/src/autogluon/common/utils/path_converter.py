import platform
from pathlib import PurePosixPath, PureWindowsPath


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
        assert not PathConverter._is_absolute(path), "It is ambiguous on how to convert an absolute path. Please provide a relative path instead"

    @staticmethod
    def to_windows(path: str) -> str:
        PathConverter._validate_path(path)
        return str(PureWindowsPath(PurePosixPath(path)))

    @staticmethod
    def to_posix(path: str) -> str:
        PathConverter._validate_path(path)
        return str(PurePosixPath(PureWindowsPath(path)))

    @staticmethod
    def to_current(path: str) -> str:
        return PathConverter.to_windows(path) if PathConverter._is_windows() else PathConverter.to_posix(path)
