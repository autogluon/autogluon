from __future__ import annotations


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


class ConstraintViolationError(AutoGluonException, AssertionError):
    """A model's fit-time constraints (`ag.max_rows`, `ag.max_features`, ...) are not satisfied.

    Signals that a model cannot be fit on this data *by configuration*, not that anything went
    wrong, so the trainer reports it as a one-line skip instead of a failure with a traceback.

    Also derives from `AssertionError`, which these constraints raised before this exception
    existed, so code catching that keeps working.

    Parameters
    ----------
    message: str
        Self-contained message, naming the model -- used when the exception surfaces on its own.
    reason: str, optional
        The same explanation without the model name, for a caller that already names the model
        (as the trainer's skip line does). Defaults to `message`.
    """

    def __init__(self, message: str, *, reason: str | None = None):
        super().__init__(message)
        self.reason = reason if reason is not None else message
