import platform
from pathlib import PurePosixPath, PureWindowsPath


class PathConverter:
    """Util class to convert a given path to a path to the corresponding OS"""

    @staticmethod
    def _is_windows():
        return platform.system() == "Windows"

    @staticmethod
    def to_windows(path: str) -> str:
        return str(PureWindowsPath(PurePosixPath(path)))

    @staticmethod
    def to_posix(path: str) -> str:
        return str(PurePosixPath(PureWindowsPath(path)))

    @staticmethod
    def to_current(path: str) -> str:
        return PathConverter.to_windows(path) if PathConverter._is_windows() else PathConverter.to_posix(path)
