class TimeLimitExceeded(Exception):
    pass


class NotEnoughMemoryError(Exception):
    pass


class NoGPUError(Exception):
    pass


class NotEnoughCudaMemoryError(Exception):
    pass


class NoValidFeatures(Exception):
    pass
