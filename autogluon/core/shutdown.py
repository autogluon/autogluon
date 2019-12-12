import logging
from ..scheduler.remote.remote_manager import RemoteManager

logger = logging.getLogger(__name__)

def done():
    """ Always call this method when you are done with all autogluon approaches.
    It will shutdown autogluon backend, such as schedulers and workers.
    We recommand calling this method in the end of your code.

    Examples
    --------
    >>> autogluon.done()
    """
    logger.info('Shutting Down AutoGluon backend.')
    RemoteManager.shutdown()
