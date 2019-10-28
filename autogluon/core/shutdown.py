from ..scheduler.remote.remote_manager import RemoteManager

def done():
    """ Always call this method when you are done with all autogluon approaches.
    It will shutdown autogluon backend, such as schedulers and workers.

    Example:
        >>> autogluon.done()
    """
    RemoteManager.shutdown()
