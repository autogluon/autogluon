from gluoncv.utils import LRSequential, LRScheduler

class LR_params:
    def __init__(self, *args):
        lr, lr_mode, num_epochs, num_batches, lr_decay_epoch, \
        lr_decay, lr_decay_period, warmup_epochs, warmup_lr= args
        self._num_batches = num_batches
        self._lr_decay = lr_decay
        self._lr_decay_period = lr_decay_period
        self._warmup_epochs = warmup_epochs
        self._warmup_lr= warmup_lr
        self._num_epochs = num_epochs
        self._lr = lr
        self._lr_mode = lr_mode
        if lr_decay_period > 0:
            lr_decay_epoch = list(range(lr_decay_period, num_epochs, lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in lr_decay_epoch.split(',')]
        self._lr_decay_epoch = [e - warmup_epochs for e in lr_decay_epoch]

        self._lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=self._warmup_lr, target_lr=lr,
                        nepochs=warmup_epochs, iters_per_epoch=self._num_batches),
            LRScheduler(lr_mode, base_lr=lr, target_lr=0,
                        nepochs=num_epochs - warmup_epochs,
                        iters_per_epoch=self._num_batches,
                        step_epoch=self._lr_decay_epoch,
                        step_factor=lr_decay, power=2)])

    @property
    def get_lr_decay_epoch(self):
        return self._lr_decay_epoch

    @property
    def get_lr_scheduler(self):
        return self._lr_scheduler