from .utils import disable_if_lite_mode


@disable_if_lite_mode(ret=1073741824)
def available_virtual_mem():
    import psutil
    return psutil.virtual_memory().available