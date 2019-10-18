import logging
from tqdm import tqdm

import mxnet as mx

from ..utils import save, load
from ..searcher import RLSearcher
from ..task.image_classification.dataset import get_built_in_dataset
from .resource import get_gpu_count, get_cpu_count
from .enas_utils import *

__all__ = ['ENAS_Scheduler']

logger = logging.getLogger(__name__)

IMAGENET_TRAINING_SAMPLES = 1281167

class ENAS_Scheduler(object):
    """ENAS Scheduler, which automatically creates LSTM controller based on the search spaces.

    """
    def __init__(self, supernet, train_set='imagenet', val_set=None,
                 train_fn=default_train_fn, eval_fn=default_val_fn,
                 train_args={}, val_args={}, num_gpus=get_gpu_count(), batch_size=256,
                 epochs=200, warmup_epochs=5, controller_lr=3.5e-4, controller_batch_size=10,
                 ema_baseline_decay=0.95, update_arch_frequency=5, **kwargs):
        ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else mx.cpu(0)
        supernet.collect_params().reset_ctx(ctx)
        supernet.hybridize()
        self.supernet = supernet
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        dataset_name = train_set
        if isinstance(train_set, str):
            train_set = get_built_in_dataset(dataset_name, train=True, batch_size=batch_size).init()
            val_set = get_built_in_dataset(dataset_name, train=False, batch_size=batch_size).init()
        if isinstance(train_set, gluon.data.Dataset):
            self.train_data = gluon.data.DataLoader(
                    train_set, batch_size=batch_size, shuffle=True,
                    last_batch="rollover", num_workers=get_cpu_count())
            self.val_data = gluon.data.DataLoader(
                    val_set, batch_size=batch_size, shuffle=True,
                    num_workers=get_cpu_count())
        else:
            self.train_data = train_set
            self.val_data = val_set
        iters_per_epoch = len(train_set) if hasattr(train_set, '__len__') else IMAGENET_TRAINING_SAMPLES // batch_size
        self.train_args = init_default_train_args(batch_size, self.supernet, epochs, iters_per_epoch) \
                if len(train_args) == 0 else train_args
        self.val_args = val_args
        self.val_args['ctx'] = ctx
        self.val_args['batch_fn'] = imagenet_batch_fn if dataset_name == 'imagenet' else default_batch_fn
        self.train_args['ctx'] = ctx
        self.train_args['batch_fn'] = imagenet_batch_fn if dataset_name == 'imagenet' else default_batch_fn
        self.ctx = ctx

        self.epochs = epochs
        self.baseline = None
        self.warmup_epochs = warmup_epochs
        # create RL searcher/controller
        self.ema_decay = ema_baseline_decay
        self.searcher = RLSearcher(self.supernet.kwspaces)
        # controller setup
        self.controller = self.searcher.controller
        self.controller_optimizer = mx.gluon.Trainer(
                self.controller.collect_params(), 'adam',
                optimizer_params={'learning_rate': controller_lr*controller_batch_size})
        self.update_arch_frequency = update_arch_frequency
        self.controller_batch_size = controller_batch_size

    def run(self):
        for epoch in tqdm(range(self.epochs)):
            # for rocord data
            if hasattr(self.train_data, 'reset'): self.train_data.reset()
            tbar = tqdm(enumerate(self.train_data))
            for i, batch in tbar:
                # sample network configuration
                config = self.controller.sample()[0]
                self.supernet.sample(**config)
                self.train_fn(self.supernet, batch, **self.train_args)
                if epoch >= self.warmup_epochs and (i % self.update_arch_frequency) == 0:
                    self.train_controller()
                tbar.set_description('epoch {}, iter {}, avg reward: {}'.format(epoch, i, self.baseline))

    def train_controller(self):
        """Run multiple number of trials
        """
        decay = self.ema_decay
        if hasattr(self.val_data, 'reset'): self.val_data.reset()
        # update 
        for i, batch in enumerate(self.val_data):
            if i >= self.controller_batch_size: break
            with mx.autograd.record():
                # sample controller_batch_size number of configurations
                configs, log_probs, entropies = self.controller.sample(batch_size=1, with_details=True)
                # schedule the training tasks and gather the reward
                self.supernet.sample(**configs[0])
                reward = self.eval_fn(self.supernet, batch, **self.val_args)
                self.baseline = reward if not self.baseline else self.baseline
                # substract baseline
                avg_rewards = mx.nd.array([reward - self.baseline],
                                          ctx=self.controller.context)
                # EMA baseline
                self.baseline = decay * self.baseline + (1 - decay) * reward
                # negative policy gradient
                log_probs = log_probs.sum(axis=1)
                loss = - log_probs * avg_rewards
                loss = loss.sum()

        # update
        loss.backward()
        self.controller_optimizer.step(self.controller_batch_size)
        logger.debug('controller loss: {}'.format(loss.asscalar()))
