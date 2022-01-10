from ctypes import *
from ctypes.util import find_library
import sys
import os
import threading
import string

__all__ = ['cudaInit', 'cudaDeviceGetCount', 'cudaSystemGetNVMLVersion',
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
