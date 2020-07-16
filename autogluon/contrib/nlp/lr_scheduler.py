import math
from mxnet import lr_scheduler


class InverseSquareRootScheduler(lr_scheduler.LRScheduler):
    """ Reduce the learning rate according to a polynomial of given power.

    During warmup
        Increase the learning rate linearly from warmup_init_lr to base_lr,
    After warmup
        Decay the learning rate with
            lr = base_lr * sqrt(warmup_steps) / sqrt(num_update)

    Parameters
    ----------
        warmup_steps
            maximum number of updates before the decay reaches final learning rate.
        base_lr
            The final learning rate in the warm-up stage. The learning rate starts to decay after
            the lr reaches warmup_end_lr
        warmup_init_lr
            The initial learning rate of the scheduler. The warm up starts at this point.
    """

    def __init__(self, warmup_steps: int, base_lr: float = 1E-3, warmup_init_lr: float = 0.0):
        super(InverseSquareRootScheduler, self).__init__(
            base_lr, warmup_steps, warmup_init_lr, 'linear')
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps

    def __call__(self, num_update):
        if num_update < self.warmup_steps:
            return self.get_warmup_lr(num_update)
        else:
            return self.base_lr * math.sqrt(self.warmup_steps) / math.sqrt(num_update)
