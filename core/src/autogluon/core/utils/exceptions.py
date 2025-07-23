
class AutoGluonException(Exception):
    """
    Generic AutoGluon exception.
    Can be used to identify AutoGluon specific exception classes.
    """
    pass


class InsufficientTime(AutoGluonException):
    """
    Similar to TimeLimitExceeded, raised when the expected outcome of an operation
    would exceed the time limit, prior to exceeding the time limit.
    """
    pass


class TimeLimitExceeded(InsufficientTime):
    """
    Exception raised when the time limit has been exceeded (over budget)
    """
    pass


class NotEnoughMemoryError(AutoGluonException):
    pass


class NoGPUError(AutoGluonException):
    pass


class NotEnoughCudaMemoryError(AutoGluonException):
    pass


class NoValidFeatures(AutoGluonException):
    pass


class NoStackFeatures(NoValidFeatures):
    pass


class NotValidStacker(AutoGluonException):
    pass
