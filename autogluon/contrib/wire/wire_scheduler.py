import mxnet as mx
from ..enas import ENAS_Scheduler

__all__ = ['Wire_Scheduler']

class Wire_Scheduler(ENAS_Scheduler):
    """ENAS Scheduler, which automatically creates LSTM controller based on the search spaces.
    """
    def __init__(self, supernet, *args, reward_fn=lambda metric, net: metric,
                 checkname='./wire_Nov1/checkpoint.ag',**kwargs):
        super(Wire_Scheduler, self).__init__(supernet, *args, reward_fn=reward_fn,
                                             checkname=checkname, **kwargs)
