from ctypes import *
from ctypes.util import find_library
import sys
import os
import threading
import string

__all__ = ['cudaInit', 'cudaDeviceGetCount', 'cudaSystemGetNVMLVersion'
           'cudaShutdown']

NVML_SUCCESS                                = 0
NVML_ERROR_UNINITIALIZED                    = 1
NVML_ERROR_LIBRARY_NOT_FOUND                = 12
NVML_ERROR_FUNCTION_NOT_FOUND               = 13
NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE        = 80

cudaLib = None
libLoadLock = threading.Lock()
_cudaLib_refcount = 0 # Incremented on each cudaInit and decremented on cudaShutdown

## C function wrappers ##
def cudaInit():
    if not _LoadNvmlLibrary():
        return False

    #
    # Initialize the library
    #
    fn = _cudaGetFunctionPointer("nvmlInit_v2")
    ret = fn()
    try:
        _cudaCheckReturn(ret)
    except NVMLError:
        return False

    # Atomically update refcount
    global _cudaLib_refcount
    libLoadLock.acquire()
    _cudaLib_refcount += 1
    libLoadLock.release()
    return True

## Device get functions
def cudaDeviceGetCount():
    c_count = c_uint()
    fn = _cudaGetFunctionPointer("nvmlDeviceGetCount_v2")
    ret = fn(byref(c_count))
    _cudaCheckReturn(ret)
    return c_count.value

def _LoadNvmlLibrary():
    '''
    Load the library if it isn't loaded already
    '''
    global cudaLib

    ret = True
    if (cudaLib == None):
        # lock to ensure only one caller loads the library
        libLoadLock.acquire()
        try:
            # ensure the library still isn't loaded
            if (cudaLib == None):
                try:
                    if (sys.platform[:3] == "win"):
                        # cdecl calling convention
                        # load cuda.dll from %ProgramFiles%/NVIDIA Corporation/NVSMI/cuda.dll
                        cudaLib = CDLL(os.path.join(os.getenv("ProgramFiles", "C:/Program Files"), "NVIDIA Corporation/NVSMI/cuda.dll"))
                    else:
                        # assume linux
                        cudaLib = CDLL("libnvidia-ml.so.1")
                except OSError as ose:
                    pass

                if (cudaLib == None):
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
    return c_version.value.decode('UTF-8')

## Function access ##
_cudaGetFunctionPointer_cache = dict() # function pointers are cached to prevent unnecessary libLoadLock locking
def _cudaGetFunctionPointer(name):
    global cudaLib

    if name in _cudaGetFunctionPointer_cache:
        return _cudaGetFunctionPointer_cache[name]

    libLoadLock.acquire()
    try:
        # ensure library was loaded
        if (cudaLib == None):
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
    if (ret != NVML_SUCCESS):
        raise NVMLError(ret)
    return ret

class NVMLError(Exception):
    _valClassMapping = dict()
    # List of currently known error codes
    _errcode_to_string = {
        NVML_ERROR_UNINITIALIZED:       "Uninitialized",
        NVML_ERROR_LIBRARY_NOT_FOUND:   "NVML Shared Library Not Found",
        }
    def __new__(typ, value):
        '''
        Maps value to a proper subclass of NVMLError.
        See _extractNVMLErrorsAsClasses function for more details
        '''
        if typ == NVMLError:
            typ = NVMLError._valClassMapping.get(value, typ)
        obj = Exception.__new__(typ)
        obj.value = value
        return obj

    def __str__(self):
        try:
            if self.value not in NVMLError._errcode_to_string:
                NVMLError._errcode_to_string[self.value] = str(cudaErrorString(self.value))
            return NVMLError._errcode_to_string[self.value]
        except Exception:
            return "NVML Error with code %d" % self.value

    def __eq__(self, other):
        return self.value == other.value

def cudaShutdown():
    #
    # Leave the library loaded, but shutdown the interface
    #
    fn = _cudaGetFunctionPointer("nvmlShutdown")
    ret = fn()
    _cudaCheckReturn(ret)

    # Atomically update refcount
    global _cudaLib_refcount
    libLoadLock.acquire()
    if (0 < _cudaLib_refcount):
        _cudaLib_refcount -= 1
    libLoadLock.release()
    return None


class _PrintableStructure(Structure):
    """
    Abstract class that produces nicer __str__ output than ctypes.Structure.
    e.g. instead of:
      >>> print str(obj)
      <class_name object at 0x7fdf82fef9e0>
    this class will print
      class_name(field_name: formatted_value, field_name: formatted_value)
    _fmt_ dictionary of <str _field_ name> -> <str format>
    e.g. class that has _field_ 'hex_value', c_uint could be formatted with
      _fmt_ = {"hex_value" : "%08X"}
    to produce nicer output.
    Default fomratting string for all fields can be set with key "<default>" like:
      _fmt_ = {"<default>" : "%d MHz"} # e.g all values are numbers in MHz.
    If not set it's assumed to be just "%s"
    Exact format of returned str from this class is subject to change in the future.
    """
    _fmt_ = {}
    def __str__(self):
        result = []
        for x in self._fields_:
            key = x[0]
            value = getattr(self, key)
            fmt = "%s"
            if key in self._fmt_:
                fmt = self._fmt_[key]
            elif "<default>" in self._fmt_:
                fmt = self._fmt_["<default>"]
            result.append(("%s: " + fmt) % (key, value))
        return self.__class__.__name__ + "(" + string.join(result, ", ") + ")"


class c_nvmlUtilization_t(_PrintableStructure):
    _fields_ = [
        ('gpu', c_uint),
        ('memory', c_uint),
    ]
    _fmt_ = {'<default>': "%d %%"}


class c_nvmlMemory_t(_PrintableStructure):
    _fields_ = [
        ('total', c_ulonglong),
        ('free', c_ulonglong),
        ('used', c_ulonglong),
    ]
    _fmt_ = {'<default>': "%d B"}

def cudaDeviceGetHandleByIndex(index):
    c_index = c_uint(index)
    device = c_nvmlDevice_t()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetHandleByIndex_v2")
    ret = fn(c_index, byref(device))
    _nvmlCheckReturn(ret)
    return device

class NviSMI:
    def __init__(self, gpu_id=0):
        self.handle = cudaDeviceGetHandleByIndex(index)

    def get_utilization_rates():
        c_util = c_nvmlUtilization_t()
        fn = _nvmlGetFunctionPointer("nvmlDeviceGetUtilizationRates")
        ret = fn(self.handle, byref(c_util))
        _nvmlCheckReturn(ret)
        return c_util

    def get_memory_info():
        c_memory = c_nvmlMemory_t()
        fn = _nvmlGetFunctionPointer("nvmlDeviceGetMemoryInfo")
        ret = fn(self.handle, byref(c_memory))
        _nvmlCheckReturn(ret)
        return c_memory


def cudaDeviceGetUtilizationRates(gpu_id=None):
    """Short summary.

    Parameters
    ----------
    gpu_id : int, list of int, or `None`
        GPU id or list of GPU id. If default to `None`, it assumes all GPU is queried.

    Returns
    -------
    dict or list of dict


    """
    gpu_id = gpu_id if gpu_id is not None else
    handle = nvmlDeviceGetHandleByIndex(0)
    c_util = c_nvmlUtilization_t()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetUtilizationRates")
    ret = fn(handle, byref(c_util))
    _nvmlCheckReturn(ret)
    return c_util

def cudaDeviceGetMemoryInfo(gpu_id=None):

    handle = handle = nvmlDeviceGetHandleByIndex(0)
    c_memory = c_nvmlMemory_t()
    fn = _nvmlGetFunctionPointer("nvmlDeviceGetMemoryInfo")
    ret = fn(handle, byref(c_memory))
    _nvmlCheckReturn(ret)
    return c_memory
