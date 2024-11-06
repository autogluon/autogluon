
class AutoGluonException(Exception):
    """
    Generic AutoGluon exception.
    Can be used to identify AutoGluon specific exception classes.
    """
    pass


class TimeLimitExceeded(AutoGluonException):
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
