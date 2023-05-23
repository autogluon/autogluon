from pathlib import PurePosixPath, PureWindowsPath

import platform


class PathConverter:
    
    @staticmethod
    def is_windows():
        return platform.system() == "Windows"
    
    @staticmethod
    def to_windows(path: str) -> str:
        return str(PureWindowsPath(PurePosixPath(path)))
    
    @staticmethod
    def to_posix(path: str) -> str:
        return str(PurePosixPath(PureWindowsPath(path)))
    
    @staticmethod
    def to_current(path: str) -> str:
        return PathConverter.to_windows(path) if PathConverter.is_windows() else PathConverter.to_posix(path)
