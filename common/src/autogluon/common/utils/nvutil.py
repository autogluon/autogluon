import atexit
import os
import sys
import threading
from ctypes import *

__all__ = [
    "ensure_initialized",
    "cudaDeviceGetCount",
    "cudaDeviceGetUUIDs",
    "cudaDeviceGetMemoryInfo",
    "cudaSystemGetNVMLVersion",
]

NVML_SUCCESS = 0
NVML_ERROR_UNINITIALIZED = 1
NVML_ERROR_LIBRARY_NOT_FOUND = 12
NVML_ERROR_FUNCTION_NOT_FOUND = 13
NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE = 80

cudaLib = None
libLoadLock = threading.Lock()


## C function wrappers ##
_initialized_pid = None


def ensure_initialized():
    """Initialize NVML for this process once; a forked child initializes again for itself.

    Returns False when the library or a driver is missing. NVML stays initialized for the life of
    the process (an init/shutdown pair costs about 15 ms while no CUDA context exists); it is shut
    down at interpreter exit.
    """
    global _initialized_pid
    if _initialized_pid == os.getpid():
        return True
    if not _LoadNvmlLibrary():
        return False
    ret = _cudaGetFunctionPointer("nvmlInit_v2")()
    if ret != NVML_SUCCESS:
        return False
    if _initialized_pid is None:
        atexit.register(_shutdown_at_exit)
    _initialized_pid = os.getpid()
    return True


def _shutdown_at_exit():
    global _initialized_pid
    if _initialized_pid == os.getpid():
        try:
            _cudaGetFunctionPointer("nvmlShutdown")()
        except NVMLError:
            pass
        _initialized_pid = None


## Device get functions
def cudaDeviceGetCount():
    c_count = c_uint()
    fn = _cudaGetFunctionPointer("nvmlDeviceGetCount_v2")
    ret = fn(byref(c_count))
    _cudaCheckReturn(ret)
    return c_count.value


def cudaDeviceGetUUIDs():
    """UUID of every device NVML enumerates, in NVML index order."""
    uuids = []
    get_handle = _cudaGetFunctionPointer("nvmlDeviceGetHandleByIndex_v2")
    get_uuid = _cudaGetFunctionPointer("nvmlDeviceGetUUID")
    for idx in range(cudaDeviceGetCount()):
        handle = c_void_p()
        _cudaCheckReturn(get_handle(c_uint(idx), byref(handle)))
        buf = create_string_buffer(96)
        _cudaCheckReturn(get_uuid(handle, buf, c_uint(96)))
        uuids.append(buf.value.decode("ascii"))
    return uuids


class _nvmlMemory_t(Structure):
    _fields_ = [("total", c_ulonglong), ("free", c_ulonglong), ("used", c_ulonglong)]


def cudaDeviceGetMemoryInfo(index):
    """`(total, free, used)` bytes of the device at NVML index `index`."""
    handle = c_void_p()
    _cudaCheckReturn(_cudaGetFunctionPointer("nvmlDeviceGetHandleByIndex_v2")(c_uint(index), byref(handle)))
    memory = _nvmlMemory_t()
    _cudaCheckReturn(_cudaGetFunctionPointer("nvmlDeviceGetMemoryInfo")(handle, byref(memory)))
    return memory.total, memory.free, memory.used


def _LoadNvmlLibrary():
    """
    Load the library if it isn't loaded already
    """
    global cudaLib

    ret = True
    if cudaLib == None:
        # lock to ensure only one caller loads the library
        libLoadLock.acquire()
        try:
            # ensure the library still isn't loaded
            if cudaLib == None:
                try:
                    if sys.platform[:3] == "win":
                        # The driver installs nvml.dll into System32; older drivers only into NVSMI.
                        candidates = [
                            os.path.join(os.getenv("SystemRoot", "C:/Windows"), "System32", "nvml.dll"),
                            os.path.join(
                                os.getenv("ProgramFiles", "C:/Program Files"),
                                "NVIDIA Corporation",
                                "NVSMI",
                                "nvml.dll",
                            ),
                        ]
                    else:
                        candidates = ["libnvidia-ml.so.1"]
                    for candidate in candidates:
                        try:
                            cudaLib = CDLL(candidate)
                            break
                        except OSError:
                            continue
                except OSError:
                    pass

                if cudaLib == None:
                    ret = False
        finally:
            # lock is always freed
            libLoadLock.release()

    return ret


def cudaSystemGetNVMLVersion():
    c_version = create_string_buffer(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE)
    fn = _cudaGetFunctionPointer("nvmlSystemGetNVMLVersion")
    ret = fn(c_version, c_uint(NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE))
    _cudaCheckReturn(ret)
    return c_version.value.decode("UTF-8")


## Function access ##
_cudaGetFunctionPointer_cache = dict()  # function pointers are cached to prevent unnecessary libLoadLock locking


def _cudaGetFunctionPointer(name):
    global cudaLib

    if name in _cudaGetFunctionPointer_cache:
        return _cudaGetFunctionPointer_cache[name]

    libLoadLock.acquire()
    try:
        # ensure library was loaded
        if cudaLib == None:
            raise NVMLError(NVML_ERROR_UNINITIALIZED)
        try:
            _cudaGetFunctionPointer_cache[name] = getattr(cudaLib, name)
            return _cudaGetFunctionPointer_cache[name]
        except AttributeError:
            raise NVMLError(NVML_ERROR_FUNCTION_NOT_FOUND)
    finally:
        # lock is always freed
        libLoadLock.release()


def _cudaCheckReturn(ret):
    if ret != NVML_SUCCESS:
        raise NVMLError(ret)
    return ret


class NVMLError(Exception):
    _errcode_to_string = {
        NVML_ERROR_UNINITIALIZED: "Uninitialized",
        NVML_ERROR_LIBRARY_NOT_FOUND: "NVML Shared Library Not Found",
        NVML_ERROR_FUNCTION_NOT_FOUND: "NVML Function Not Found",
    }

    def __init__(self, value):
        super().__init__(value)
        self.value = value

    def __str__(self):
        if self.value in NVMLError._errcode_to_string:
            return NVMLError._errcode_to_string[self.value]
        try:
            if not _LoadNvmlLibrary():
                raise NVMLError(NVML_ERROR_LIBRARY_NOT_FOUND)
            fn = _cudaGetFunctionPointer("nvmlErrorString")
            fn.restype = c_char_p
            return fn(c_uint(self.value)).decode("utf-8")
        except Exception:
            return "NVML Error with code %d" % self.value

    def __eq__(self, other):
        return isinstance(other, NVMLError) and self.value == other.value

    def __hash__(self):
        return hash(self.value)
